from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
MODEL_ORDER = ["KNN", "LogisticRegression", "SVM", "RandomForest", "ExtraTrees", "GBDT", "MLP"]


def make_model(name: str, class_weight: str | None = None) -> Pipeline:
    median = ("imputer", SimpleImputer(strategy="median"))
    scaler = ("scaler", StandardScaler())

    if name == "KNN":
        if class_weight is not None:
            raise ValueError("KNN does not support class_weight.")
        steps = [median, scaler, ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2))]
    elif name == "LogisticRegression":
        steps = [median, scaler, ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0, random_state=RANDOM_STATE, class_weight=class_weight))]
    elif name == "SVM":
        steps = [median, scaler, ("clf", SVC(C=10.0, kernel="rbf", gamma="scale", random_state=RANDOM_STATE, class_weight=class_weight))]
    elif name == "RandomForest":
        steps = [median, ("clf", RandomForestClassifier(n_estimators=600, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1, class_weight=class_weight))]
    elif name == "ExtraTrees":
        steps = [median, ("clf", ExtraTreesClassifier(n_estimators=600, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1, class_weight=class_weight))]
    elif name == "GBDT":
        if class_weight is not None:
            raise ValueError("The configured GBDT model does not support class_weight.")
        steps = [median, ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE))]
    elif name == "MLP":
        if class_weight is not None:
            raise ValueError("The configured sklearn MLP does not support class_weight.")
        steps = [median, scaler, ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", alpha=0.001, learning_rate_init=0.001, max_iter=1500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=40, random_state=RANDOM_STATE))]
    else:
        raise KeyError(f"Unknown model: {name}")
    return Pipeline(steps)


def model_parameter_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        for parameter, value in make_model(name).get_params(deep=True).items():
            if parameter.startswith("clf__"):
                rows.append({"Model": name, "Parameter": parameter.removeprefix("clf__"), "Value": repr(value)})
    return rows

