from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import combinations

import pandas as pd


TYPE_COL = "Type"
CLASS_ORDER = ["A", "S", "I"]
IMPUTATION_METHODS = ["global_mean", "knn"]
N_OUTER_FOLDS = 5
RHO_TAG = "0.90"

NON_FEATURE_COLS = {
    "No.", "No", "Samp1e", "Sample", "Sample_ID", "SampleID", "ID",
    "Reference", "Type", "Type-1", "Type-2",
}

# These registries reproduce the pairwise-ratio construction rule in Step 01.
# Membership in the generated set, rather than a substring or prefix test, is
# the sole source of truth for identifying systematically constructed ratios.
MAJOR_RATIO_INPUTS = (
    "SiO2(wt%)", "TiO2", "Al2O3", "Fe2O3t", "MgO",
    "CaO", "Na2O", "K2O", "MnO", "P2O5",
)
TRACE_RATIO_INPUTS = (
    "Ga(ppm)", "Rb", "Sr", "Y", "Zr", "Nb", "Ba", "La", "Ce", "Pr",
    "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "Pb", "Th", "U", "Cs",
)


def _ratio_registry(inputs: tuple[str, ...], prefix: str, domain: str) -> dict[str, dict[str, str]]:
    return {
        f"{prefix}{numerator}/{denominator}": {
            "Ratio_domain": domain,
            "Numerator": numerator,
            "Denominator": denominator,
        }
        for numerator, denominator in combinations(inputs, 2)
    }


SYSTEMATIC_RATIO_REGISTRY = {
    **_ratio_registry(MAJOR_RATIO_INPUTS, "R_Major_", "major"),
    **_ratio_registry(TRACE_RATIO_INPUTS, "R_Trace_", "trace_ree"),
}

# One conceptual classical index followed by its exact column aliases. Alias
# order is preference order. Raw Sr/Y is deliberately preferred over the
# mathematically duplicated R_Trace_Sr/Y column. Exact aliases also guarantee
# that R_Trace_Sr/Yb cannot be confused with Sr/Y.
CLASSICAL_FEATURE_ALIASES = (
    ("A/CNK", ("A/CNK",)),
    ("A/NK", ("A/NK",)),
    ("10000*Ga/Al", ("10000*Ga/Al", "10000*Ga/A1", "10000×Ga/Al")),
    ("Zr+Nb+Ce+Y", ("Zr+Nb+Ce+Y",)),
    ("Nb/Ta", ("R_Trace_Nb/Ta", "Nb/Ta")),
    ("Zr/Hf", ("R_Trace_Zr/Hf", "Zr/Hf")),
    ("Sr/Y", ("Sr/Y", "R_Trace_Sr/Y")),
    ("Rb/Sr", ("R_Trace_Rb/Sr", "Rb/Sr")),
    ("Fe2O3t/MgO", ("R_Major_Fe2O3t/MgO", "Fe2O3t/MgO")),
    ("Na2O/K2O", ("R_Major_Na2O/K2O", "Na2O/K2O")),
)

_CLASSICAL_CONCEPT_BY_EXACT_ALIAS = {
    alias: concept
    for concept, aliases in CLASSICAL_FEATURE_ALIASES
    for alias in aliases
}


def normalize_type_value(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "A-type": "A", "A-Type": "A", "A_TYPE": "A", "A type": "A",
        "S-type": "S", "S-Type": "S", "S_TYPE": "S", "S type": "S",
        "I-type": "I", "I-Type": "I", "I_TYPE": "I", "I type": "I",
    }
    return aliases.get(text, text.upper() if text.lower() in {"a", "s", "i"} else text)


def display_feature_name(name: object) -> str:
    text = str(name)
    return (
        text.replace("R_Major_", "")
        .replace("R_Trace_", "")
        .replace("A12O3", "Al2O3")
        .replace("10000*Ga/A1", "10000×Ga/Al")
        .replace("10000*Ga/Al", "10000×Ga/Al")
    )


def feature_key(name: object) -> str:
    text = display_feature_name(name).replace("×", "*")
    return re.sub(r"\s+", "", text).lower()


def exact_feature_key(name: object) -> str:
    text = str(name).strip().replace("A12O3", "Al2O3")
    text = text.replace("10000×Ga/Al", "10000*Ga/Al").replace("10000*Ga/A1", "10000*Ga/Al")
    return re.sub(r"\s+", "", text).lower()


def is_constructed_ratio(name: object) -> bool:
    """Return registry membership; never infer origin from a substring."""
    return str(name).strip() in SYSTEMATIC_RATIO_REGISTRY


def is_classical_feature(name: object) -> bool:
    """Return exact classical-alias membership; Sr/Yb is therefore excluded."""
    return str(name).strip() in _CLASSICAL_CONCEPT_BY_EXACT_ALIAS


def is_candidate_novel_ratio(name: object) -> bool:
    text = str(name).strip()
    return text in SYSTEMATIC_RATIO_REGISTRY and text not in _CLASSICAL_CONCEPT_BY_EXACT_ALIAS


def ratio_group(name: object) -> str:
    text = str(name).strip()
    meta = SYSTEMATIC_RATIO_REGISTRY.get(text)
    if meta is None:
        return "Original/classical feature"
    if meta["Ratio_domain"] == "major":
        return "Major-element ratio"

    parts = {meta["Numerator"], meta["Denominator"]}
    hfse = {"Nb", "Ta", "Zr", "Hf", "Ti", "Y"}
    ree = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
    lile = {"Rb", "Sr", "Ba", "Cs", "Pb"}
    th_u = {"Th", "U"}
    if parts & hfse and parts & ree:
        return "HFSE-REE ratio"
    if parts & hfse and parts & th_u:
        return "HFSE-Th/U ratio"
    if parts & lile and parts & ree:
        return "LILE-REE ratio"
    if parts & lile and parts & th_u:
        return "LILE-Th/U ratio"
    if parts <= ree:
        return "REE-REE ratio"
    return "Other trace-element ratio"


def resolve_one_feature(feature: object, columns: Iterable[object]) -> str | None:
    """Resolve an exact feature name only; no prefix-stripping equivalence."""
    text = str(feature).strip()
    column_names = [str(column).strip() for column in columns]
    return text if text in column_names else None


def resolve_feature_list(features: Iterable[object], columns: Iterable[object]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    missing: list[str] = []
    for feature in features:
        match = resolve_one_feature(feature, columns)
        if match is None:
            missing.append(str(feature))
        elif match not in resolved:
            resolved.append(match)
    return resolved, missing


def get_classical_features(columns: Iterable[object]) -> list[str]:
    column_names = {str(column).strip() for column in columns}
    selected: list[str] = []
    for concept, aliases in CLASSICAL_FEATURE_ALIASES:
        matches = [alias for alias in aliases if alias in column_names]
        if not matches:
            raise ValueError(f"Missing classical feature concept {concept!r}; expected one of {aliases}")
        selected.append(matches[0])
    return selected


def build_feature_metadata(columns: Iterable[object]) -> pd.DataFrame:
    """Create the explicit, auditable metadata table for candidate columns."""
    column_names = [str(column).strip() for column in columns if str(column).strip() not in NON_FEATURE_COLS]
    preferred_classical = set(get_classical_features(column_names))
    rows: list[dict[str, object]] = []
    for feature in column_names:
        ratio_meta = SYSTEMATIC_RATIO_REGISTRY.get(feature)
        concept = _CLASSICAL_CONCEPT_BY_EXACT_ALIAS.get(feature, "")
        is_systematic = ratio_meta is not None
        rows.append(
            {
                "Feature": feature,
                "Display_feature": display_feature_name(feature),
                "Feature_origin": "systematic_pairwise_ratio" if is_systematic else "non_systematic",
                "Is_systematic_ratio": is_systematic,
                "Ratio_domain": ratio_meta["Ratio_domain"] if ratio_meta else "",
                "Numerator": ratio_meta["Numerator"] if ratio_meta else "",
                "Denominator": ratio_meta["Denominator"] if ratio_meta else "",
                "Classical_concept": concept,
                "Is_classical_alias": bool(concept),
                "Is_preferred_classical_feature": feature in preferred_classical,
                "Include_in_non_ratio_baseline": not is_systematic,
                "Is_fold_novel_candidate": is_systematic and not bool(concept),
            }
        )
    metadata = pd.DataFrame(rows)
    if metadata["Feature"].duplicated().any():
        duplicates = metadata.loc[metadata["Feature"].duplicated(), "Feature"].tolist()
        raise ValueError(f"Duplicate feature names in candidate matrix: {duplicates}")
    return metadata


def build_fold_feature_sets(columns: Iterable[object], champions: Iterable[object]) -> dict[str, list[str]]:
    column_names = [str(column).strip() for column in columns if str(column).strip() not in NON_FEATURE_COLS]
    metadata = build_feature_metadata(column_names)
    resolved_champions, missing = resolve_feature_list(champions, column_names)
    if missing:
        raise ValueError(f"Fold champion features missing from fold data: {missing[:20]}")

    non_ratio = metadata.loc[metadata["Include_in_non_ratio_baseline"], "Feature"].tolist()
    if len(column_names) == 443 and len(non_ratio) != 47:
        raise ValueError(f"Expected 47 non-ratio baseline features among 443 candidates; found {len(non_ratio)}")
    novel_champions = metadata.loc[
        metadata["Feature"].isin(resolved_champions) & metadata["Is_fold_novel_candidate"],
        "Feature",
    ].tolist()
    if not novel_champions:
        raise ValueError("No fold-specific systematic novel-ratio champions were found.")

    return {
        "Non_ratio_baseline": non_ratio,
        "Non_ratio_plus_fold_novel": list(dict.fromkeys(non_ratio + novel_champions)),
        "Full_candidate_features": column_names,
        "Fold_cluster_champions": resolved_champions,
    }
