#!/usr/bin/env python3
# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Train GPBoost admission prediction model (v2).

Architecture: LightGBM gradient boosting + per-program random intercepts.
Handles missing data natively. Replaces the per-program LR in train_model.py.

Usage:
    python tools/train_model_v2.py
    python tools/train_model_v2.py --evaluate-only
    python tools/train_model_v2.py --output data/models/admission_model_v2.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import gpboost as gpb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FEATURE_MATRIX = _PROJECT_ROOT / "data" / "admissions" / "feature_matrix.csv"
_PROGRAM_YAML_DIR = _PROJECT_ROOT / "data" / "programs"
_OUTPUT_MODEL = _PROJECT_ROOT / "data" / "models" / "admission_model_v2.json"

# Features to use
FEATURE_COLS = [
    "gpa_normalized",
    "gre_quant",
    "undergrad_tier_encoded",
    "intern_score",
    "research_score",
    "is_international",
    "is_female",
    "major_relevance_score",
]

# Missing indicator columns (added during preprocessing)
MISSING_INDICATOR_COLS = [
    "has_gpa",
    "has_gre",
    "has_tier",
    "has_intern",
    "has_nationality",
]

# Real acceptance rates from program YAMLs (for bias correction)
REAL_ACCEPT_RATES = {
    "baruch-mfe": 0.04, "princeton-mfin": 0.054, "cmu-mscf": 0.172,
    "columbia-msfe": 0.133, "mit-mfin": 0.083, "berkeley-mfe": 0.174,
    "uchicago-msfm": 0.22, "gatech-qcf": 0.302, "columbia-mafn": 0.223,
    "ncstate-mfm": 0.167, "cornell-mfe": 0.207, "nyu-courant": 0.225,
    "nyu-tandon-mfe": 0.381, "ucla-mfe": 0.36, "fordham-msqf": 0.594,
    "uwash-cfrm": 0.539, "uiuc-msfe": 0.507, "rutgers-mqf": 0.864,
    "stevens-mfe": 0.68, "uminn-mfm": 0.807, "bu-msmf": 0.806,
    "jhu-mfm": 0.507, "stanford-mcf": 0.05, "umich-mfe": 0.30,
    "northwestern-mfe": 0.30, "usc-msmf": 0.40, "utoronto-mmf": 0.30,
    "toronto-mmf": 0.30,  # alternate name in training data
    "columbia-msfe-econ": 0.05,  # Columbia Business School MS Financial Economics
    "yale-am": 0.08,  # Yale MA Applied Mathematics
    "stanford-mse": 0.10,  # Stanford MS Management Science & Engineering
}

# Program name to ID mapping (from feature_matrix)
PROGRAM_ID_MAP: dict[str, int] = {}
PROGRAM_NAME_MAP: dict[int, str] = {}


def load_data() -> pd.DataFrame:
    """Build feature matrix directly from training_data_full.csv.

    This avoids the program_id encoding mismatch between
    prepare_training_data.py's feature_matrix.csv and our own encoding.
    We do our own encoding here so program names are always known.
    """
    global PROGRAM_ID_MAP, PROGRAM_NAME_MAP

    full_path = _PROJECT_ROOT / "data" / "admissions" / "training_data_full.csv"
    df = pd.read_csv(full_path)
    df = df.replace({"": np.nan, "nan": np.nan})

    # Filter to records with valid result
    df = df[df["result"].isin(["accepted", "rejected", "waitlisted"])].copy()
    df["result_binary"] = (df["result"] == "accepted").astype(int)

    # Build program encoding (our own, consistent)
    programs = sorted(df["program"].dropna().unique())
    PROGRAM_ID_MAP = {name: i for i, name in enumerate(programs)}
    PROGRAM_NAME_MAP = {i: name for name, i in PROGRAM_ID_MAP.items()}
    df["program_id"] = df["program"].map(PROGRAM_ID_MAP).fillna(-1).astype(int)

    # Encode features
    df["gpa_normalized"] = pd.to_numeric(df["gpa"], errors="coerce")
    df["gre_quant"] = pd.to_numeric(df["gre_quant"], errors="coerce")

    # Undergrad tier encoding
    tier_map = {"T10": 1, "C9": 1, "T20": 2, "985": 2, "T30": 3, "211": 3,
                "T50": 4, "top_intl": 2, "intl": 4, "other_cn": 5, "other": 5}
    df["undergrad_tier_encoded"] = df["undergrad_tier"].map(tier_map)

    # Intern score
    intern_map = {"us_top_quant": 10, "us_quant": 8, "us_bb": 7, "us_finance": 6,
                  "cn_top": 6, "us_tech": 5, "cn_finance": 4, "other": 2, "none": 0}
    df["intern_score"] = df["intern_level"].map(intern_map)

    # Research score
    research_map = {"published": 3, "significant": 2, "minor": 1, "none": 0}
    df["research_score"] = df["research_level"].map(research_map).fillna(0)

    # International flag
    nat_intl = {"chinese": 1, "indian": 1, "korean": 1, "other_asian": 1,
                "european": 1, "other": 1, "us": 0}
    df["is_international"] = df["nationality"].map(nat_intl)

    # Gender
    df["is_female"] = (df["gender"] == "F").astype(float)
    df.loc[df["gender"].isna() | (df["gender"] == "unknown"), "is_female"] = np.nan

    # Major relevance
    df["major_relevance_score"] = pd.to_numeric(df.get("major_relevance"), errors="coerce")

    # Missing indicators
    df["has_gpa"] = (~df["gpa_normalized"].isna()).astype(float)
    df["has_gre"] = (~df["gre_quant"].isna()).astype(float)
    df["has_tier"] = (~df["undergrad_tier_encoded"].isna()).astype(float)
    df["has_intern"] = (~df["intern_score"].isna()).astype(float)
    df["has_nationality"] = (~df["is_international"].isna()).astype(float)

    print(f"  Program map: {len(PROGRAM_ID_MAP)} programs")
    return df


def build_program_maps(df: pd.DataFrame) -> None:
    """Build program ID ↔ name mappings from the ACTUAL feature matrix.

    The feature_matrix.csv has integer program_ids assigned by
    prepare_training_data.py. We need to recover the name→id mapping
    from training_data_model.csv which has both program name and the
    same integer encoding.
    """
    global PROGRAM_ID_MAP, PROGRAM_NAME_MAP

    model_path = _PROJECT_ROOT / "data" / "admissions" / "training_data_model.csv"
    fm_path = _PROJECT_ROOT / "data" / "admissions" / "feature_matrix.csv"

    # Reconstruct the EXACT encoding used by prepare_training_data.py:
    # It reads training_data_full.csv programs, gets canonical names,
    # sorts them, and assigns integer IDs alphabetically.
    # We replicate that logic here.
    full_path = _PROJECT_ROOT / "data" / "admissions" / "training_data_full.csv"
    if full_path.exists():
        _full_df = pd.read_csv(full_path, usecols=["program"], dtype=str)  # noqa: F841
        # Get unique programs from feature_matrix (which has the actual IDs used)
        fm_df = pd.read_csv(fm_path, usecols=["program_id"])
        _actual_pids = sorted(fm_df["program_id"].dropna().unique())  # noqa: F841

        # The encoding in prepare_training_data.py is:
        # sorted(set(PROGRAM_CANONICAL.values())) → enumerate
        # We don't have PROGRAM_CANONICAL here, but we can recover it:
        # Read training_data_model.csv which has program names, and
        # feature_matrix which has program_id. They share the same source rows.
        if model_path.exists():
            model_df = pd.read_csv(model_path, usecols=["program"], dtype=str)
            # Build map from unique (program, count) to figure out pid assignment
            programs_in_model = sorted(model_df["program"].dropna().unique())
            # The encoding is: sorted unique canonical names → 0,1,2,...
            PROGRAM_ID_MAP = {name: i for i, name in enumerate(programs_in_model)}
            PROGRAM_NAME_MAP = {i: name for name, i in PROGRAM_ID_MAP.items()}

    print(f"  Program map: {len(PROGRAM_ID_MAP)} programs")


def train_gpboost(df: pd.DataFrame) -> tuple:
    """Train the GPBoost model."""
    feature_cols = FEATURE_COLS + MISSING_INDICATOR_COLS
    X = df[feature_cols].values
    y = df["result_binary"].values
    group_data = df["program_id"].values.reshape(-1, 1)

    # Create GPModel with random intercept per program
    gp_model = gpb.GPModel(
        group_data=group_data,
        likelihood="bernoulli_logit",
    )

    # Conservative parameters for ~10K samples
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 4,
    }

    dataset = gpb.Dataset(X, y)

    print("Training GPBoost model...")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Programs: {len(np.unique(group_data))}")
    print(f"  Accept rate: {y.mean():.1%}")

    bst = gpb.train(
        params=params,
        train_set=dataset,
        gp_model=gp_model,
        num_boost_round=300,
    )

    return bst, gp_model, feature_cols


def evaluate_cv(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """Run stratified K-fold cross-validation."""
    feature_cols = FEATURE_COLS + MISSING_INDICATOR_COLS
    X = df[feature_cols].values
    y = df["result_binary"].values
    group_data = df["program_id"].values.reshape(-1, 1)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = group_data[train_idx]
        g_test = group_data[test_idx]

        gp_model = gpb.GPModel(
            group_data=g_train,
            likelihood="bernoulli_logit",
        )

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }

        bst = gpb.train(
            params=params,
            train_set=gpb.Dataset(X_train, y_train),
            gp_model=gp_model,
            num_boost_round=300,
        )

        # Predict
        pred = bst.predict(
            data=X_test,
            group_data_pred=g_test,
            predict_var=False,
            pred_latent=False,
        )
        # GPBoost predict returns dict with 'response_mean'
        if isinstance(pred, dict):
            probs = pred["response_mean"]
        else:
            probs = pred

        probs = np.clip(probs, 0.001, 0.999)
        all_preds[test_idx] = probs

        # Metrics
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        fold_metrics.append({"fold": fold + 1, "auc": auc, "brier": brier, "n": len(y_test)})
        print(f"  Fold {fold + 1}: AUC={auc:.4f}  Brier={brier:.4f}  n={len(y_test)}")

    # Overall metrics
    overall_auc = roc_auc_score(y, all_preds)
    overall_brier = brier_score_loss(y, all_preds)
    avg_auc = np.mean([m["auc"] for m in fold_metrics])
    avg_brier = np.mean([m["brier"] for m in fold_metrics])

    print(f"\n  Overall AUC:   {overall_auc:.4f} (avg across folds: {avg_auc:.4f})")
    print(f"  Overall Brier: {overall_brier:.4f} (avg across folds: {avg_brier:.4f})")

    # Per-program metrics (top programs)
    program_ids = df["program_id"].values
    print("\n  Per-Program Metrics (programs with 50+ samples):")
    for pid in sorted(np.unique(program_ids)):
        mask = program_ids == pid
        if mask.sum() < 50:
            continue
        y_p = y[mask]
        p_p = all_preds[mask]
        if len(np.unique(y_p)) < 2:
            continue
        auc_p = roc_auc_score(y_p, p_p)
        brier_p = brier_score_loss(y_p, p_p)
        name = PROGRAM_NAME_MAP.get(pid, str(pid))
        print(f"    {name:25} AUC={auc_p:.3f}  Brier={brier_p:.3f}  n={mask.sum()}")

    # Feature importance
    return {
        "overall_auc": overall_auc,
        "overall_brier": overall_brier,
        "avg_auc": avg_auc,
        "avg_brier": avg_brier,
        "fold_metrics": fold_metrics,
        "predictions": all_preds.tolist(),
    }


def compute_bias_corrections(
    bst, gp_model, df: pd.DataFrame
) -> dict[str, float]:
    """Compute per-program bias corrections using real acceptance rates."""
    corrections = {}

    # Build name map directly (don't rely on global state)
    name_map = {pid: name for name, pid in PROGRAM_ID_MAP.items()}

    # Get learned random effects
    sorted_pids = sorted(df["program_id"].unique())
    random_effects = gp_model.predict(
        group_data_pred=np.array([[pid] for pid in sorted_pids]),
    )
    if isinstance(random_effects, dict):
        re_means = random_effects["mu"]
    else:
        re_means = random_effects

    for i, pid in enumerate(sorted_pids):
        name = name_map.get(pid, PROGRAM_NAME_MAP.get(pid, ""))
        if name in REAL_ACCEPT_RATES:
            real_rate = REAL_ACCEPT_RATES[name]
            # Training accept rate for this program
            mask = df["program_id"] == pid
            train_rate = df.loc[mask, "result_binary"].mean()

            # Correction: shift random intercept
            logit_real = np.log(real_rate / (1 - real_rate))
            logit_train = np.log(train_rate / (1 - train_rate))
            correction = logit_real - logit_train

            corrections[name] = {
                "real_rate": real_rate,
                "train_rate": round(train_rate, 4),
                "correction": round(correction, 4),
                "learned_re": round(float(re_means[i]) if i < len(re_means) else 0, 4),
            }

    return corrections


def save_model(
    bst,
    gp_model,
    feature_cols: list[str],
    cv_results: dict,
    corrections: dict,
    output_path: Path,
) -> None:
    """Save model to JSON for the inference engine."""
    # Feature importance
    importance = bst.feature_importance(importance_type="gain")
    feature_importance = {
        col: round(float(imp), 2)
        for col, imp in zip(feature_cols, importance)
    }

    model_data = {
        "model_version": 2,
        "architecture": "gpboost_mixed_effects",
        "description": "LightGBM gradient boosting + per-program random intercepts",
        "features": feature_cols,
        "feature_importance": feature_importance,
        "cv_metrics": {
            "auc": round(cv_results["overall_auc"], 4),
            "brier_score": round(cv_results["overall_brier"], 4),
            "n_folds": len(cv_results["fold_metrics"]),
            "fold_details": cv_results["fold_metrics"],
        },
        "bias_corrections": corrections,
        "real_accept_rates": REAL_ACCEPT_RATES,
        "program_id_map": {name: pid for name, pid in PROGRAM_ID_MAP.items()},
        "training_stats": {
            "n_samples": len(cv_results["predictions"]),
            "n_programs": len(PROGRAM_ID_MAP),
            "accept_rate": round(
                np.mean([1 if p > 0.5 else 0 for p in cv_results["predictions"]]),
                4,
            ),
        },
    }

    # Save the GPBoost model binary separately
    model_bin_path = output_path.with_suffix(".bin")
    bst.save_model(str(model_bin_path))
    model_data["model_binary_path"] = str(model_bin_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model_data, f, indent=2, default=str)

    print(f"\n  Model saved to: {output_path}")
    print(f"  Binary saved to: {model_bin_path}")
    print("  Feature importance:")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp / max(importance) * 20)
        print(f"    {feat:30} {imp:8.1f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPBoost admission model v2")
    parser.add_argument("--output", "-o", default=str(_OUTPUT_MODEL))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  QuantPath Model v2: GPBoost Training Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    # Program maps built inside load_data()
    print(f"  Records: {len(df)}")
    print(f"  Programs: {df['program_id'].nunique()}")
    print(f"  Accept/Reject: {(df['result_binary']==1).sum()}/{(df['result_binary']==0).sum()}")

    # Cross-validation
    print(f"\n{'─'*60}")
    print("5-Fold Stratified Cross-Validation:")
    print(f"{'─'*60}")
    cv_results = evaluate_cv(df)

    if args.evaluate_only:
        return

    # Train final model on all data
    print(f"\n{'─'*60}")
    print("Training Final Model (all data):")
    print(f"{'─'*60}")
    bst, gp_model, feature_cols = train_gpboost(df)

    # Bias corrections
    print(f"\n{'─'*60}")
    print("Computing Bias Corrections:")
    print(f"{'─'*60}")
    corrections = compute_bias_corrections(bst, gp_model, df)
    for name, corr in sorted(corrections.items(), key=lambda x: x[1]["real_rate"]):
        r = corr["real_rate"]
        t = corr["train_rate"]
        s = corr["correction"]
        print(f"  {name:25} real={r:.1%}  train={t:.1%}  shift={s:+.2f}")

    # Save
    print(f"\n{'─'*60}")
    print("Saving Model:")
    print(f"{'─'*60}")
    save_model(bst, gp_model, feature_cols, cv_results, corrections, Path(args.output))

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"  AUC: {cv_results['overall_auc']:.4f}  Brier: {cv_results['overall_brier']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
