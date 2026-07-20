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

from granite_ml.config import (  # noqa: E402
    CHAMPION_DIR,
    FOLDS_DIR,
    RAW_DATA_FILE,
    RESULTS_DIR,
    STABILITY_DIR,
)

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 0. 配置：通常只需要修改 PROJECT_ROOT
# ============================================================

INPUT_ROOT = FOLDS_DIR

OUT_ROOT = CHAMPION_DIR

TYPE_COL = "Type"
CLASS_ORDER = ["A", "S", "I"]

NON_FEATURE_COLS = {
    "No.",
    "Samp1e",
    "Sample",
    "Type",
    "Type-1",
    "Type-2",
    "Reference",
}

IMPUTATION_METHODS = ["global_mean", "knn"]
N_OUTER_FOLDS = 5
N_INNER_SPLITS = 5
SEED = 42

# 这些阈值只做相关结构诊断，不逐一查看外层测试表现。
RHO_DIAGNOSTIC_LIST = [0.75, 0.80, 0.85, 0.90, 0.95]

# 正式建模只使用这一个阈值。
MODEL_RHO = 0.90

# 每个 inner validation fold 中记录 SHAP Top-K。
TOPK = 50

# 是否保存每个 rho 的簇大小图。
SAVE_CLUSTER_SIZE_PLOTS = True

# 443×443 的相关矩阵较大，默认不写入每个工作簿。
SAVE_SPEARMAN_CORR_IN_WORKBOOK = False

# GPU 可用时设为 True；否则保持 False。
USE_GPU = False

XGB_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.75,
    gamma=0.0,
    reg_lambda=6.0,
    reg_alpha=0.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)

if USE_GPU:
    XGB_PARAMS["device"] = "cuda"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("INPUT_ROOT:", INPUT_ROOT)
print("OUT_ROOT:", OUT_ROOT)
print("MODEL_RHO:", MODEL_RHO)
print("Ablation: disabled")

# ============================================================
# 1. 通用工具与输入检查
# ============================================================

def safe_rho_tag(rho: float) -> str:
    return f"{float(rho):.2f}"


def build_model() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def normalize_type_value(value) -> str:
    text = str(value).strip()
    mapping = {
        "A-type": "A",
        "A-Type": "A",
        "A_TYPE": "A",
        "A type": "A",
        "A型": "A",
        "S-type": "S",
        "S-Type": "S",
        "S_TYPE": "S",
        "S type": "S",
        "S型": "S",
        "I-type": "I",
        "I-Type": "I",
        "I_TYPE": "I",
        "I type": "I",
        "I型": "I",
    }
    return mapping.get(text, text)


def feature_complexity(name: str) -> tuple[int, int]:
    text = str(name)
    operator_count = sum(
        text.count(symbol)
        for symbol in ["+", "-", "*", "/", "(", ")", "^"]
    )
    return operator_count, len(text)


def is_ratio_feature(name: str) -> bool:
    text = str(name)
    return (
        text.startswith("R_Major_")
        or text.startswith("R_Trace_")
    )


def find_fold_file(
    method: str,
    outer_fold: int,
    split: str,
) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split 必须是 train 或 test。")

    method_dir = INPUT_ROOT / method
    expected = (
        method_dir
        / f"fold_{outer_fold:02d}_{split}_with_ratios.xlsx"
    )

    if expected.exists():
        return expected

    patterns = [
        f"*fold*{outer_fold:02d}*{split}*ratio*.xlsx",
        f"*fold*{outer_fold}*{split}*ratio*.xlsx",
        f"*{outer_fold:02d}*{split}*.xlsx",
    ]

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(method_dir.glob(pattern))

    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"未找到 {method} outer fold {outer_fold} 的 "
            f"{split} 文件。预期路径：{expected}"
        )

    raise RuntimeError(
        f"{method} outer fold {outer_fold} 的 {split} 文件不唯一："
        f"{[str(p) for p in candidates]}"
    )


def read_fold_data(
    method: str,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    train_path = find_fold_file(method, outer_fold, "train")
    test_path = find_fold_file(method, outer_fold, "test")

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    if TYPE_COL not in train_df.columns:
        raise ValueError(f"训练文件缺少标签列 {TYPE_COL}: {train_path}")
    if TYPE_COL not in test_df.columns:
        raise ValueError(f"测试文件缺少标签列 {TYPE_COL}: {test_path}")

    train_df[TYPE_COL] = train_df[TYPE_COL].map(
        normalize_type_value
    )
    test_df[TYPE_COL] = test_df[TYPE_COL].map(
        normalize_type_value
    )

    return train_df, test_df, train_path, test_path


def find_sample_id_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str | None:
    candidates = ["No.", "Sample", "Samp1e"]
    for column in candidates:
        if column not in train_df.columns or column not in test_df.columns:
            continue
        if not train_df[column].is_unique:
            continue
        if not test_df[column].is_unique:
            continue
        return column
    return None


def check_train_test_separation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    id_col = find_sample_id_column(train_df, test_df)
    if id_col is None:
        print(
            "  提示：未找到可唯一识别样品的ID列，"
            "跳过train/test样品ID重叠检查。"
        )
        return

    train_ids = set(train_df[id_col].astype(str))
    test_ids = set(test_df[id_col].astype(str))
    overlap = train_ids.intersection(test_ids)

    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"训练集和测试集存在 {len(overlap)} 个重复样品ID，"
            f"列={id_col}，示例={examples}"
        )


def get_feature_columns_from_train(
    train_df: pd.DataFrame,
) -> list[str]:
    candidate_columns = [
        c
        for c in train_df.columns
        if c not in NON_FEATURE_COLS
    ]

    numeric_train = train_df[candidate_columns].copy()
    for column in numeric_train.columns:
        numeric_train[column] = pd.to_numeric(
            numeric_train[column],
            errors="coerce",
        )

    numeric_train = numeric_train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # 所有删除判断只基于当前 outer-train。
    numeric_train = numeric_train.loc[
        :,
        numeric_train.notna().any(axis=0),
    ]
    numeric_train = numeric_train.loc[
        :,
        numeric_train.nunique(dropna=True) > 1,
    ]

    return numeric_train.columns.tolist()


def prepare_train_test_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    LabelEncoder,
    list[str],
]:
    check_train_test_separation(train_df, test_df)
    feature_columns = get_feature_columns_from_train(train_df)

    if not feature_columns:
        raise ValueError("当前 outer-train 没有可用数值特征。")

    missing_in_test = [
        c for c in feature_columns
        if c not in test_df.columns
    ]
    if missing_in_test:
        raise ValueError(
            "以下训练特征在测试文件中不存在："
            f"{missing_in_test[:20]}"
        )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    for column in feature_columns:
        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )
        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce",
        )

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    train_nan = int(X_train.isna().sum().sum())
    test_nan = int(X_test.isna().sum().sum())

    if train_nan > 0:
        bad = X_train.columns[
            X_train.isna().any(axis=0)
        ].tolist()
        raise ValueError(
            f"X_train 中仍有 {train_nan} 个 NaN/inf。"
            f"请检查01折内预处理。涉及列：{bad[:20]}"
        )

    if test_nan > 0:
        bad = X_test.columns[
            X_test.isna().any(axis=0)
        ].tolist()
        raise ValueError(
            f"X_test 中仍有 {test_nan} 个 NaN/inf。"
            f"请检查01折内预处理。涉及列：{bad[:20]}"
        )

    y_train_raw = train_df[TYPE_COL].astype(str).to_numpy()
    y_test_raw = test_df[TYPE_COL].astype(str).to_numpy()

    unexpected_labels = sorted(
        set(y_train_raw)
        .union(set(y_test_raw))
        .difference(CLASS_ORDER)
    )
    if unexpected_labels:
        raise ValueError(
            f"发现预期 A/S/I 之外的标签：{unexpected_labels}"
        )

    missing_train_classes = sorted(
        set(CLASS_ORDER).difference(y_train_raw)
    )
    if missing_train_classes:
        raise ValueError(
            f"当前训练折缺少类别：{missing_train_classes}"
        )

    # 固定为 A、S、I 顺序，保证所有折的类别报告和混淆矩阵一致。
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(CLASS_ORDER, dtype=object)
    y_train = encoder.transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train,
        y_test,
        encoder,
        feature_columns,
    )

# ============================================================
# 2. Spearman 相关聚类
# ============================================================

class UnionFind:
    def __init__(self, n_items: int):
        self.parent = list(range(n_items))
        self.rank = [0] * n_items

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[
                self.parent[item]
            ]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def compute_spearman_corr(X_train: pd.DataFrame) -> pd.DataFrame:
    # 只允许传入当前 outer-train。
    corr = X_train.corr(method="spearman")
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def spearman_clusters_from_corr(
    corr: pd.DataFrame,
    rho_threshold: float,
) -> list[list[str]]:
    columns = list(corr.columns)
    values = corr.to_numpy()
    n_features = len(columns)

    union_find = UnionFind(n_features)

    for i in range(n_features):
        right_part = values[i, i + 1 :]
        connected_offsets = np.where(
            np.abs(right_part) >= rho_threshold
        )[0]

        for offset in connected_offsets:
            j = i + 1 + int(offset)
            union_find.union(i, j)

    grouped: dict[int, list[str]] = {}
    for i, feature in enumerate(columns):
        root = union_find.find(i)
        grouped.setdefault(root, []).append(feature)

    clusters = list(grouped.values())
    clusters.sort(
        key=lambda features: (
            -len(features),
            str(features[0]),
        )
    )
    return clusters


def count_high_corr_pairs(
    corr: pd.DataFrame,
    rho_threshold: float,
) -> int:
    values = np.abs(corr.to_numpy())
    upper = values[
        np.triu_indices_from(values, k=1)
    ]
    return int(np.sum(upper >= rho_threshold))


def cluster_membership_table(
    clusters: list[list[str]],
) -> pd.DataFrame:
    rows = []
    for cluster_id, features in enumerate(clusters, start=1):
        for feature in features:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(features),
                    "feature": feature,
                }
            )
    return pd.DataFrame(rows)

# ============================================================
# 3. Inner-CV XGBoost + SHAP
# ============================================================

def mean_abs_shap_by_feature(
    model: XGBClassifier,
    X_validation: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_output = explainer.shap_values(X_validation)
    n_features = X_validation.shape[1]

        # Compatibility with SHAP releases that return one matrix per class.
    if isinstance(shap_output, list):
        class_arrays = [
            np.asarray(values)
            for values in shap_output
        ]
        stacked = np.stack(class_arrays, axis=0)
        result = np.mean(np.abs(stacked), axis=(0, 1))
        return np.asarray(result, dtype=float)

    # 兼容新版 SHAP Explanation。
    if hasattr(shap_output, "values"):
        shap_output = shap_output.values

    values = np.asarray(shap_output)

    if values.ndim == 2:
        if values.shape[1] != n_features:
            raise ValueError(
                f"无法识别 SHAP 形状：{values.shape}"
            )
        return np.mean(np.abs(values), axis=0)

    if values.ndim == 3:
        feature_axes = [
            axis
            for axis, size in enumerate(values.shape)
            if size == n_features
        ]

        # 常见形状为 (samples, features, classes)。
        if 1 in feature_axes:
            return np.mean(np.abs(values), axis=(0, 2))

        # 兼容 (classes, samples, features)。
        if 2 in feature_axes:
            return np.mean(np.abs(values), axis=(0, 1))

    raise ValueError(
        f"无法识别多分类 SHAP 输出形状：{values.shape}"
    )


def compute_inner_cv_shap_scores(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[
    pd.DataFrame,
    list[float],
    list[tuple[np.ndarray, np.ndarray]],
]:
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )
    inner_splits = list(inner_cv.split(X_train, y_train))

    n_features = X_train.shape[1]
    importance_by_fold = np.zeros(
        (N_INNER_SPLITS, n_features),
        dtype=float,
    )
    topk_count = np.zeros(n_features, dtype=int)
    inner_full_f1 = []

    for inner_fold, (train_idx, valid_idx) in enumerate(
        inner_splits,
        start=1,
    ):
        X_inner_train = X_train.iloc[train_idx]
        X_inner_valid = X_train.iloc[valid_idx]
        y_inner_train = y_train[train_idx]
        y_inner_valid = y_train[valid_idx]

        model = build_model()
        model.fit(X_inner_train, y_inner_train)

        prediction = model.predict(X_inner_valid)
        fold_f1 = f1_score(
            y_inner_valid,
            prediction,
            average="macro",
            zero_division=0,
        )
        inner_full_f1.append(float(fold_f1))

        fold_importance = mean_abs_shap_by_feature(
            model,
            X_inner_valid,
        )

        if len(fold_importance) != n_features:
            raise ValueError(
                "SHAP重要性长度与特征数不一致："
                f"{len(fold_importance)} vs {n_features}"
            )

        importance_by_fold[inner_fold - 1] = fold_importance

        k = min(TOPK, n_features)
        top_indices = np.argsort(-fold_importance)[:k]
        topk_count[top_indices] += 1

        print(
            f"    inner fold {inner_fold}/{N_INNER_SPLITS}: "
            f"full-feature Macro-F1={fold_f1:.4f}"
        )

    importance_mean = importance_by_fold.mean(axis=0)
    importance_std = importance_by_fold.std(
        axis=0,
        ddof=1,
    )
    topk_frequency = topk_count / N_INNER_SPLITS

    missing_rate = X_train.isna().mean(axis=0)
    zero_rate = (X_train == 0).mean(axis=0)

    score_table = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance_mean_shap": importance_mean,
            "importance_std_shap": importance_std,
            "topk_count": topk_count,
            "topk_freq_ratio": topk_frequency,
            "score(importance×stability)": (
                importance_mean * topk_frequency
            ),
            "missing_rate": [
                float(missing_rate.get(f, 0.0))
                for f in X_train.columns
            ],
            "zero_rate": [
                float(zero_rate.get(f, 0.0))
                for f in X_train.columns
            ],
            "is_ratio": [
                is_ratio_feature(f)
                for f in X_train.columns
            ],
            "complex_ops": [
                feature_complexity(f)[0]
                for f in X_train.columns
            ],
            "name_len": [
                feature_complexity(f)[1]
                for f in X_train.columns
            ],
        }
    )

    score_table = score_table.sort_values(
        "score(importance×stability)",
        ascending=False,
    ).reset_index(drop=True)

    return score_table, inner_full_f1, inner_splits

# ============================================================
# 4. 每个相关簇选择一个 champion
# ============================================================

def select_cluster_champions(
    clusters: list[list[str]],
    feature_score_table: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    score_map = (
        feature_score_table
        .set_index("feature")
        .to_dict(orient="index")
    )

    champions: list[str] = []
    rows = []

    for cluster_id, cluster_features in enumerate(
        clusters,
        start=1,
    ):
        candidates = []

        for feature in cluster_features:
            if feature not in score_map:
                continue

            values = score_map[feature]
            importance = float(
                values["importance_mean_shap"]
            )
            frequency = float(
                values["topk_freq_ratio"]
            )
            score = importance * frequency
            complexity, name_length = feature_complexity(feature)

            candidates.append(
                {
                    "feature": feature,
                    "importance_mean": importance,
                    "importance_std": float(
                        values["importance_std_shap"]
                    ),
                    "topk_count": int(
                        values["topk_count"]
                    ),
                    "topk_freq_ratio": frequency,
                    "score": score,
                    "missing_rate": float(
                        values["missing_rate"]
                    ),
                    "zero_rate": float(
                        values["zero_rate"]
                    ),
                    "complex_ops": complexity,
                    "name_len": name_length,
                    "is_ratio": is_ratio_feature(feature),
                }
            )

        if not candidates:
            continue

        # 先按 SHAP×稳定频率最大化；若并列，优先缺失/零值少、
        # 形式更简单且名称更短的特征。
        candidates.sort(
            key=lambda row: (
                -row["score"],
                -row["importance_mean"],
                -row["topk_freq_ratio"],
                row["missing_rate"],
                row["zero_rate"],
                row["complex_ops"],
                row["name_len"],
                row["feature"],
            )
        )

        champion = candidates[0]
        champions.append(champion["feature"])

        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_features),
                "champion": champion["feature"],
                "champion_score": champion["score"],
                "champion_importance_mean": (
                    champion["importance_mean"]
                ),
                "champion_importance_std": (
                    champion["importance_std"]
                ),
                "champion_topk_count": (
                    champion["topk_count"]
                ),
                "champion_topk_freq_ratio": (
                    champion["topk_freq_ratio"]
                ),
                "champion_missing_rate": (
                    champion["missing_rate"]
                ),
                "champion_zero_rate": (
                    champion["zero_rate"]
                ),
                "champion_is_ratio": (
                    champion["is_ratio"]
                ),
                "champion_complex_ops": (
                    champion["complex_ops"]
                ),
                "champion_name_len": (
                    champion["name_len"]
                ),
            }
        )

    champion_table = pd.DataFrame(rows)
    champion_table = champion_table.sort_values(
        "champion_score",
        ascending=False,
    ).reset_index(drop=True)

    return champions, champion_table


def evaluate_inner_feature_subset(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_subset: list[str],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, list[float]]:
    fold_scores = []

    for train_idx, valid_idx in inner_splits:
        model = build_model()
        model.fit(
            X_train.iloc[train_idx][feature_subset],
            y_train[train_idx],
        )
        prediction = model.predict(
            X_train.iloc[valid_idx][feature_subset]
        )
        fold_score = f1_score(
            y_train[valid_idx],
            prediction,
            average="macro",
            zero_division=0,
        )
        fold_scores.append(float(fold_score))

    return (
        float(np.mean(fold_scores)),
        float(np.std(fold_scores, ddof=1)),
        fold_scores,
    )

# ============================================================
# 5. 外层测试评价
# ============================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "Balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "Macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_and_evaluate_outer(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: list[str],
    encoder: LabelEncoder,
    feature_set_name: str,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    if not features:
        raise ValueError(
            f"{feature_set_name} 的特征列表为空。"
        )

    model = build_model()
    model.fit(X_train[features], y_train)
    prediction = model.predict(X_test[features])

    metrics: dict[str, object] = {
        "Feature_set": feature_set_name,
        "N_features": len(features),
        "Features": "; ".join(features),
    }
    metrics.update(evaluate_predictions(y_test, prediction))

    class_report = classification_report(
        y_test,
        prediction,
        labels=list(range(len(encoder.classes_))),
        target_names=list(encoder.classes_),
        output_dict=True,
        zero_division=0,
    )

    class_rows = []
    for class_name in encoder.classes_:
        values = class_report[str(class_name)]
        class_rows.append(
            {
                "Feature_set": feature_set_name,
                "Class": str(class_name),
                "Precision": float(values["precision"]),
                "Recall": float(values["recall"]),
                "F1": float(values["f1-score"]),
                "Support": int(values["support"]),
            }
        )

    matrix = confusion_matrix(
        y_test,
        prediction,
        labels=list(range(len(encoder.classes_))),
    )
    confusion = pd.DataFrame(
        matrix,
        index=[
            f"True_{name}"
            for name in encoder.classes_
        ],
        columns=[
            f"Pred_{name}"
            for name in encoder.classes_
        ],
    )

    return metrics, pd.DataFrame(class_rows), confusion

# ============================================================
# 6. 绘图
# ============================================================

def plot_cluster_sizes(
    clusters: list[list[str]],
    rho_threshold: float,
    output_path: Path,
    title_prefix: str,
) -> None:
    cluster_sizes = [len(cluster) for cluster in clusters]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=1000)
    ax.hist(cluster_sizes, bins=30)
    ax.set_title(
        f"{title_prefix}: cluster sizes "
        f"(|rho| >= {rho_threshold:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_rho_structure_sensitivity(
    sensitivity_table: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
) -> None:
    table = sensitivity_table.sort_values("rho_th")

    fig, left_axis = plt.subplots(
        figsize=(9, 5.5),
        dpi=1000,
    )

    left_axis.plot(
        table["rho_th"],
        table["n_high_corr_pairs"],
        marker="o",
        color="tab:blue",
        label="Highly correlated pairs",
    )
    left_axis.set_xlabel("Spearman threshold |rho|")
    left_axis.set_ylabel(
        "No. of highly correlated pairs",
        color="tab:blue",
    )
    left_axis.tick_params(axis="y", labelcolor="tab:blue")

    right_axis = left_axis.twinx()
    right_axis.plot(
        table["rho_th"],
        table["n_champions"],
        marker="s",
        color="tab:orange",
        label="Clusters / champions",
    )
    right_axis.set_ylabel(
        "No. of clusters / champions",
        color="tab:orange",
    )
    right_axis.tick_params(
        axis="y",
        labelcolor="tab:orange",
    )

    left_axis.set_title(
        f"{title_prefix}: rho structural sensitivity",
        fontsize=13,
        fontweight="bold",
    )
    left_axis.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 7. 保存一个 rho 的兼容工作簿
# ============================================================

def save_rho_workbook(
    method: str,
    outer_fold: int,
    rho_threshold: float,
    rho_directory: Path,
    feature_score_table: pd.DataFrame,
    champion_table: pd.DataFrame,
    membership_table: pd.DataFrame,
    performance_values: dict[str, float],
    corr: pd.DataFrame,
) -> Path:
    rho_tag = safe_rho_tag(rho_threshold)
    output_path = (
        rho_directory
        / (
            f"outer_fold_{outer_fold:02d}_"
            f"cluster_champions_SHAP_innerCV_"
            f"rho{rho_tag}.xlsx"
        )
    )

    run_configuration = pd.DataFrame(
        {
            "Parameter": [
                "Method",
                "Outer_fold",
                "rho_th",
                "MODEL_RHO",
                "Ablation_performed",
                "Feature_selection_scope",
                "Outer_test_used_for_rho_selection",
                "Downstream_performance_feature_sheet",
            ],
            "Value": [
                method,
                outer_fold,
                rho_threshold,
                MODEL_RHO,
                False,
                "Current outer-training partition only",
                False,
                "ClusterChampions",
            ],
        }
    )

    performance_table = pd.DataFrame(
        {
            "Metric": list(performance_values.keys()),
            "Value": list(performance_values.values()),
        }
    )

    downstream_features = champion_table[
        ["champion"]
    ].rename(columns={"champion": "Feature"})
    downstream_features.insert(
        0,
        "Feature_set",
        "Fold_cluster_champions",
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        feature_score_table.to_excel(
            writer,
            index=False,
            sheet_name="FeatureScores_SHAP_innerCV",
        )
        champion_table.to_excel(
            writer,
            index=False,
            sheet_name="ClusterChampions",
        )
        membership_table.to_excel(
            writer,
            index=False,
            sheet_name="ClusterMembership",
        )
        downstream_features.to_excel(
            writer,
            index=False,
            sheet_name="DownstreamFeatures",
        )
        performance_table.to_excel(
            writer,
            index=False,
            sheet_name="PerformanceSummary",
        )
        run_configuration.to_excel(
            writer,
            index=False,
            sheet_name="RunConfiguration",
        )

        if SAVE_SPEARMAN_CORR_IN_WORKBOOK:
            corr.to_excel(
                writer,
                sheet_name="SpearmanCorr_trainOnly",
            )

    return output_path

# ============================================================
# 8. 单个 method + outer fold 主流程
# ============================================================

def process_one_outer_fold(
    method: str,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(
        f"\n========== {method} | "
        f"outer fold {outer_fold} =========="
    )

    (
        train_df,
        test_df,
        train_path,
        test_path,
    ) = read_fold_data(method, outer_fold)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        feature_columns,
    ) = prepare_train_test_xy(train_df, test_df)

    print("  train file:", train_path)
    print("  test file:", test_path)
    print(
        f"  samples: train={len(X_train)}, "
        f"test={len(X_test)}"
    )
    print(f"  candidate features={len(feature_columns)}")
    print(f"  classes={encoder.classes_.tolist()}")

    fold_output_directory = (
        OUT_ROOT
        / method
        / f"outer_fold_{outer_fold:02d}"
    )
    fold_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 所有以下数据依赖步骤都只使用当前 outer-train。
    corr = compute_spearman_corr(X_train)

    print("  Computing inner-CV XGBoost-SHAP scores...")
    (
        feature_score_table,
        inner_full_f1_by_fold,
        inner_splits,
    ) = compute_inner_cv_shap_scores(X_train, y_train)

    inner_full_mean = float(
        np.mean(inner_full_f1_by_fold)
    )
    inner_full_std = float(
        np.std(inner_full_f1_by_fold, ddof=1)
    )

    sensitivity_rows = []
    outer_metric_rows = []
    classwise_tables = []
    confusion_tables: dict[str, pd.DataFrame] = {}

    # Full-feature外层评价只做一次，不随 rho 重复。
    (
        full_metrics,
        full_classwise,
        full_confusion,
    ) = train_and_evaluate_outer(
        X_train,
        y_train,
        X_test,
        y_test,
        features=list(X_train.columns),
        encoder=encoder,
        feature_set_name="Full_features",
    )

    full_metrics.update(
        {
            "Method": method,
            "Outer_fold": outer_fold,
            "rho_th": MODEL_RHO,
        }
    )
    full_classwise["Method"] = method
    full_classwise["Outer_fold"] = outer_fold
    full_classwise["rho_th"] = MODEL_RHO
    outer_metric_rows.append(full_metrics)
    classwise_tables.append(full_classwise)
    confusion_tables["Full_features"] = full_confusion

    for rho_threshold in RHO_DIAGNOSTIC_LIST:
        rho_tag = safe_rho_tag(rho_threshold)
        rho_directory = (
            fold_output_directory
            / f"rho_{rho_tag}"
        )
        rho_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        clusters = spearman_clusters_from_corr(
            corr,
            rho_threshold,
        )
        high_corr_pairs = count_high_corr_pairs(
            corr,
            rho_threshold,
        )
        champions, champion_table = (
            select_cluster_champions(
                clusters,
                feature_score_table,
            )
        )
        membership_table = cluster_membership_table(
            clusters
        )

        cluster_sizes = [
            len(cluster)
            for cluster in clusters
        ]

        inner_champion_mean = np.nan
        inner_champion_std = np.nan
        outer_full_f1 = np.nan
        outer_champion_f1 = np.nan

        # 正式模型评价只对 MODEL_RHO 执行。
        if np.isclose(rho_threshold, MODEL_RHO):
            print(
                f"  rho={rho_tag}: evaluating "
                "fold-specific cluster champions..."
            )

            (
                inner_champion_mean,
                inner_champion_std,
                _,
            ) = evaluate_inner_feature_subset(
                X_train,
                y_train,
                champions,
                inner_splits,
            )

            (
                champion_metrics,
                champion_classwise,
                champion_confusion,
            ) = train_and_evaluate_outer(
                X_train,
                y_train,
                X_test,
                y_test,
                features=champions,
                encoder=encoder,
                feature_set_name=(
                    f"Fold_cluster_champions_rho{rho_tag}"
                ),
            )

            champion_metrics.update(
                {
                    "Method": method,
                    "Outer_fold": outer_fold,
                    "rho_th": rho_threshold,
                }
            )
            champion_classwise["Method"] = method
            champion_classwise["Outer_fold"] = outer_fold
            champion_classwise["rho_th"] = rho_threshold

            outer_metric_rows.append(champion_metrics)
            classwise_tables.append(champion_classwise)
            confusion_tables[
                f"Champions_rho_{rho_tag}"
            ] = champion_confusion

            outer_full_f1 = float(
                full_metrics["Macro_F1"]
            )
            outer_champion_f1 = float(
                champion_metrics["Macro_F1"]
            )

        performance_values = {
            "inner_full_f1_mean": inner_full_mean,
            "inner_full_f1_std": inner_full_std,
            "inner_champion_f1_mean": (
                inner_champion_mean
            ),
            "inner_champion_f1_std": (
                inner_champion_std
            ),
            "outer_full_macro_f1": outer_full_f1,
            "outer_champion_macro_f1": (
                outer_champion_f1
            ),
        }

        workbook_path = save_rho_workbook(
            method=method,
            outer_fold=outer_fold,
            rho_threshold=rho_threshold,
            rho_directory=rho_directory,
            feature_score_table=feature_score_table,
            champion_table=champion_table,
            membership_table=membership_table,
            performance_values=performance_values,
            corr=corr,
        )

        if SAVE_CLUSTER_SIZE_PLOTS:
            plot_cluster_sizes(
                clusters,
                rho_threshold,
                (
                    rho_directory
                    / (
                        f"outer_fold_{outer_fold:02d}_"
                        f"cluster_size_distribution_"
                        f"rho{rho_tag}.png"
                    )
                ),
                title_prefix=f"{method} fold {outer_fold}",
            )

        sensitivity_rows.append(
            {
                "Method": method,
                "Outer_fold": outer_fold,
                "rho_th": float(rho_threshold),
                "is_model_rho": bool(
                    np.isclose(rho_threshold, MODEL_RHO)
                ),
                "n_features_total": int(
                    X_train.shape[1]
                ),
                "n_high_corr_pairs": high_corr_pairs,
                "n_clusters": int(len(clusters)),
                "n_champions": int(len(champions)),
                "reduction_ratio(champ/total)": float(
                    len(champions) / X_train.shape[1]
                ),
                "cluster_size_min": int(
                    np.min(cluster_sizes)
                ),
                "cluster_size_median": float(
                    np.median(cluster_sizes)
                ),
                "cluster_size_max": int(
                    np.max(cluster_sizes)
                ),
                "inner_full_f1_mean": inner_full_mean,
                "inner_full_f1_std": inner_full_std,
                "inner_champion_f1_mean": (
                    inner_champion_mean
                ),
                "inner_champion_f1_std": (
                    inner_champion_std
                ),
                "outer_full_macro_f1": outer_full_f1,
                "outer_champion_macro_f1": (
                    outer_champion_f1
                ),
                "result_file": str(workbook_path),
            }
        )

        print(
            f"  rho={rho_tag}: "
            f"pairs={high_corr_pairs}, "
            f"clusters/champions={len(champions)}"
        )

    sensitivity_table = pd.DataFrame(
        sensitivity_rows
    ).sort_values("rho_th")

    sensitivity_path = (
        fold_output_directory
        / (
            f"outer_fold_{outer_fold:02d}_"
            "rho_sensitivity_summary.xlsx"
        )
    )
    sensitivity_table.to_excel(
        sensitivity_path,
        index=False,
    )

    plot_rho_structure_sensitivity(
        sensitivity_table,
        (
            fold_output_directory
            / (
                f"outer_fold_{outer_fold:02d}_"
                "rho_structural_sensitivity.png"
            )
        ),
        title_prefix=f"{method} fold {outer_fold}",
    )

    outer_metrics_table = pd.DataFrame(
        outer_metric_rows
    )
    classwise_table = pd.concat(
        classwise_tables,
        axis=0,
        ignore_index=True,
    )

    metrics_path = (
        fold_output_directory
        / (
            f"outer_fold_{outer_fold:02d}_"
            "outer_test_metrics.xlsx"
        )
    )

    with pd.ExcelWriter(
        metrics_path,
        engine="openpyxl",
    ) as writer:
        outer_metrics_table.to_excel(
            writer,
            index=False,
            sheet_name="OuterTestMetrics",
        )
        classwise_table.to_excel(
            writer,
            index=False,
            sheet_name="ClasswiseMetrics",
        )
        for name, table in confusion_tables.items():
            table.to_excel(
                writer,
                sheet_name=name[:31],
            )

    return (
        sensitivity_table,
        outer_metrics_table,
        classwise_table,
    )

# ============================================================
# 9. 跨外层折汇总
# ============================================================

def summarize_across_outer_folds(
    all_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        "n_high_corr_pairs",
        "n_clusters",
        "n_champions",
        "reduction_ratio(champ/total)",
        "cluster_size_median",
        "cluster_size_max",
        "inner_full_f1_mean",
        "inner_champion_f1_mean",
        "outer_full_macro_f1",
        "outer_champion_macro_f1",
    ]

    rows = []
    for (method, rho), group in all_sensitivity.groupby(
        ["Method", "rho_th"],
        sort=True,
    ):
        row: dict[str, object] = {
            "Method": method,
            "rho_th": rho,
            "is_model_rho": bool(
                np.isclose(rho, MODEL_RHO)
            ),
            "N_outer_folds": int(
                group["Outer_fold"].nunique()
            ),
        }

        for column in numeric_columns:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            ).dropna()

            row[f"{column}_mean"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )
            row[f"{column}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def save_run_manifest() -> None:
    manifest = {
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUT_ROOT),
        "imputation_methods": IMPUTATION_METHODS,
        "n_outer_folds": N_OUTER_FOLDS,
        "n_inner_splits": N_INNER_SPLITS,
        "rho_diagnostic_list": RHO_DIAGNOSTIC_LIST,
        "model_rho": MODEL_RHO,
        "topk": TOPK,
        "seed": SEED,
        "ablation_performed": False,
        "outer_test_used_for_rho_selection": False,
        "downstream_feature_source": (
            "Per-method/per-outer-fold ClusterChampions"
        ),
        "xgb_params": XGB_PARAMS,
    }

    manifest_path = OUT_ROOT / "04_run_manifest.json"
    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

# ============================================================
# 10. 主程序
# ============================================================

def main() -> None:
    all_sensitivity_tables = []
    all_outer_metric_tables = []
    all_classwise_tables = []

    save_run_manifest()

    for method in IMPUTATION_METHODS:
        method_input_directory = INPUT_ROOT / method
        if not method_input_directory.exists():
            raise FileNotFoundError(
                f"未找到输入目录：{method_input_directory}"
            )

        for outer_fold in range(1, N_OUTER_FOLDS + 1):
            (
                sensitivity_table,
                outer_metrics_table,
                classwise_table,
            ) = process_one_outer_fold(
                method,
                outer_fold,
            )

            all_sensitivity_tables.append(
                sensitivity_table
            )
            all_outer_metric_tables.append(
                outer_metrics_table
            )
            all_classwise_tables.append(
                classwise_table
            )

    all_sensitivity = pd.concat(
        all_sensitivity_tables,
        axis=0,
        ignore_index=True,
    )
    all_outer_metrics = pd.concat(
        all_outer_metric_tables,
        axis=0,
        ignore_index=True,
    )
    all_classwise = pd.concat(
        all_classwise_tables,
        axis=0,
        ignore_index=True,
    )

    all_sensitivity.to_excel(
        OUT_ROOT
        / "all_outer_folds_rho_sensitivity_raw.xlsx",
        index=False,
    )
    all_outer_metrics.to_excel(
        OUT_ROOT
        / "all_outer_folds_outer_test_metrics_raw.xlsx",
        index=False,
    )
    all_classwise.to_excel(
        OUT_ROOT
        / "all_outer_folds_classwise_metrics_raw.xlsx",
        index=False,
    )

    summary = summarize_across_outer_folds(
        all_sensitivity
    )
    summary.to_excel(
        OUT_ROOT
        / "summary_across_outer_folds_by_method_and_rho.xlsx",
        index=False,
    )

    model_rho_rows = summary[
        np.isclose(summary["rho_th"], MODEL_RHO)
    ]

    print("\n========== 全部完成 ==========")
    print("输出目录：", OUT_ROOT)
    print(
        "后续05/06应读取每折 rho_0.90 下的 "
        "ClusterChampions。"
    )
    print(
        "全局稳定特征只用于完成外层评价后的解释，"
        "不能放回原五折计算性能。"
    )
    print("\nrho=0.90 汇总：")
    display_columns = [
        "Method",
        "n_champions_mean",
        "n_champions_std",
        "outer_champion_macro_f1_mean",
        "outer_champion_macro_f1_std",
    ]
    existing_columns = [
        c
        for c in display_columns
        if c in model_rho_rows.columns
    ]
    print(model_rho_rows[existing_columns])

if __name__ == "__main__":
    main()
