#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICML Stability Suite
===================
A single, reproducible evaluation script to quantify:
  (A) Dataset-edition divergence (e.g., MIMIC-IV demo vs full)
  (B) Cross-domain instability (e.g., Synthetic ↔ NHANES)
  (C) Stability-aware model selection (accuracy, calibration, robustness, ranking stability)

Design goals
-----------
- One command runs end-to-end: load → preprocess → CV → metrics → stability → artifacts.
- Leakage-safe: preprocessing and optional calibration are fit *inside* each CV split.
- Stability-first reporting: per-env metrics + ranking stability + pairwise preference flips.
- Publication-friendly outputs: CSV/JSON + plots + a human-readable summary.txt.

This script intentionally uses only standard scientific Python + scikit-learn.

Example
-------
# (1) MIMIC edition divergence
python icml_stability_suite.py --task mimic_mortality \
  --mimic-demo demo_analytic_dataset_mortality_all_admissions.csv \
  --mimic-full full_analytic_dataset_mortality_all_admissions.csv \
  --outdir runs/mimic

# (2) Synthetic ↔ NHANES cross-domain
python icml_stability_suite.py --task tg_high_response \
  --synthetic Synthetic_Dataset_1500_Patients_precise.csv \
  --nhanes nhanes_rsce_dataset_clean.csv \
  --label-mode nhanes_quantile_shared --label-quantile 0.75 \
  --outdir runs/tg

Notes
-----
- If your column names differ, use --target-col / --id-col / --drop-cols.
- For ICML-quality: run multi-metric + sensitivity analyses:
    --rank-score auroc --also-scores auprc logloss
    --threshold-scan 80 300 10   (for TG task, scans shared threshold)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    ECE (equal-width bins on [0,1]).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes core metrics. Assumes binary labels {0,1}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    # Guard against degenerate folds
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "logloss": np.nan,
            "brier": np.nan,
            "ece": np.nan,
        }
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, np.c_[1 - y_prob, y_prob], labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }


def bootstrap_ci(
    values: Sequence[float],
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for a 1D list of scalar values.
    """
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(np.mean(sample))
    lo = np.quantile(boots, alpha / 2.0)
    hi = np.quantile(boots, 1.0 - alpha / 2.0)
    return (float(lo), float(hi))


# -----------------------------
# Ranking / Stability
# -----------------------------

@dataclass
class RankingStability:
    ref_env: str
    score: str
    margin: float
    global_flip_rate: float
    preference_consistency: float
    spearman_mean: float


def _rankdata(x: np.ndarray) -> np.ndarray:
    """
    Rank with average ties (ascending). Higher metric => better.
    """
    x = np.asarray(x, dtype=float)
    # Convert to ranks (1..n), handle ties
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for k, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman via rank correlation. Returns nan if degenerate.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def preference_matrix(
    scores: Dict[str, float],
    margin: float
) -> pd.DataFrame:
    """
    Pairwise preference: pref[i,j] = 1 if model i beats model j by > margin, else 0.
    """
    models = list(scores.keys())
    mat = np.zeros((len(models), len(models)), dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                continue
            si = scores[mi]
            sj = scores[mj]
            if np.isnan(si) or np.isnan(sj):
                mat[i, j] = np.nan
            else:
                mat[i, j] = 1.0 if (si - sj) > margin else 0.0
    return pd.DataFrame(mat, index=models, columns=models)


def flip_fraction(
    pref_ref: pd.DataFrame,
    pref_other: pd.DataFrame
) -> float:
    """
    Fraction of pairwise comparisons that flip between two envs.
    Ignores diagonal and nan entries.
    """
    A = pref_ref.values
    B = pref_other.values
    mask = ~np.isnan(A) & ~np.isnan(B)
    # remove diagonal
    n = A.shape[0]
    for i in range(n):
        mask[i, i] = False
    if not np.any(mask):
        return float("nan")
    return float(np.mean(A[mask] != B[mask]))


def model_flip_involvement(
    pref_ref: pd.DataFrame,
    pref_other: pd.DataFrame
) -> pd.Series:
    """
    For each model, what fraction of its pairwise edges flip?
    """
    models = pref_ref.index.tolist()
    A = pref_ref.values
    B = pref_other.values
    n = len(models)
    out = {}
    for i, m in enumerate(models):
        mask = ~np.isnan(A[i, :]) & ~np.isnan(B[i, :])
        mask[i] = False
        if not np.any(mask):
            out[m] = float("nan")
        else:
            out[m] = float(np.mean(A[i, mask] != B[i, mask]))
    return pd.Series(out).sort_values()


# -----------------------------
# Data loading / task schemas
# -----------------------------

@dataclass
class TaskSpec:
    name: str
    envs: List[str]
    target_col: str
    id_col: Optional[str]
    drop_cols: List[str]
    positive_label: int = 1


def infer_task_spec(args: argparse.Namespace) -> TaskSpec:
    if args.task == "mimic_mortality":
        # Default target in many MIMIC analytic tables is 'mortality' or 'hospital_expire_flag'
        target = args.target_col or "mortality"
        return TaskSpec(
            name="mimic_mortality",
            envs=["demo", "full"],
            target_col=target,
            id_col=args.id_col,
            drop_cols=args.drop_cols or [],
        )
    if args.task == "tg_high_response":
        # We'll build a binary label from TG in NHANES and TG4h in synthetic
        return TaskSpec(
            name="tg_high_response",
            envs=["synthetic", "nhanes"],
            target_col="__LABEL__",  # constructed
            id_col=args.id_col,
            drop_cols=args.drop_cols or [],
        )
    raise ValueError(f"Unknown task: {args.task}")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names (keep original too)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_tg_labels(
    df_synth: pd.DataFrame,
    df_nhanes: pd.DataFrame,
    synth_col: str,
    nhanes_col: str,
    mode: str,
    q: float,
    shared_threshold: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Create a binary label for tg_high_response with several harmonization modes.

    mode options:
      - nhanes_quantile_shared: threshold = quantile(q) of NHANES column; apply to both
      - synth_quantile_shared:  threshold = quantile(q) of Synthetic column; apply to both (only if units compatible)
      - shared_absolute:         user-provided --shared-threshold
      - env_specific_quantile:   each env uses its own quantile(q) (NOT a shared label definition; for sensitivity)
      - cdf_match:               map synthetic TG4h to NHANES TG via CDF matching, then apply NHANES quantile threshold
    """
    meta: Dict[str, Any] = {
        "synth_col_used": synth_col,
        "nhanes_col_used": nhanes_col,
        "quantile": q,
        "mode": mode,
    }

    s = df_synth.copy()
    n = df_nhanes.copy()

    # Basic sanity
    for col, df, name in [(synth_col, s, "synthetic"), (nhanes_col, n, "nhanes")]:
        if col not in df.columns:
            raise KeyError(f"[{name}] missing column '{col}'. Available: {list(df.columns)[:30]}...")

    def qthr(df: pd.DataFrame, col: str) -> float:
        return float(df[col].dropna().quantile(q))

    if mode == "shared_absolute":
        if shared_threshold is None:
            raise ValueError("--shared-threshold is required for label-mode shared_absolute")
        thr = float(shared_threshold)
        s["__LABEL__"] = (s[synth_col] >= thr).astype(int)
        n["__LABEL__"] = (n[nhanes_col] >= thr).astype(int)
        meta["shared_threshold_used"] = thr

    elif mode == "nhanes_quantile_shared":
        thr = qthr(n, nhanes_col)
        s["__LABEL__"] = (s[synth_col] >= thr).astype(int)
        n["__LABEL__"] = (n[nhanes_col] >= thr).astype(int)
        meta["thr_from_nhanes_quantile"] = thr
        meta["shared_threshold_used"] = thr

    elif mode == "synth_quantile_shared":
        thr = qthr(s, synth_col)
        s["__LABEL__"] = (s[synth_col] >= thr).astype(int)
        n["__LABEL__"] = (n[nhanes_col] >= thr).astype(int)
        meta["thr_from_synth_quantile"] = thr
        meta["shared_threshold_used"] = thr

    elif mode == "env_specific_quantile":
        thr_s = qthr(s, synth_col)
        thr_n = qthr(n, nhanes_col)
        s["__LABEL__"] = (s[synth_col] >= thr_s).astype(int)
        n["__LABEL__"] = (n[nhanes_col] >= thr_n).astype(int)
        meta["thr_from_synth_quantile"] = thr_s
        meta["thr_from_nhanes_quantile"] = thr_n

    elif mode == "cdf_match":
        # Map synthetic values to NHANES scale by matching CDFs:
        # x_mapped = F_n^{-1}(F_s(x))
        # Then apply NHANES quantile threshold on mapped values.
        x = s[synth_col].to_numpy()
        x_valid = x[~np.isnan(x)]
        y = n[nhanes_col].to_numpy()
        y_valid = y[~np.isnan(y)]
        if len(x_valid) < 10 or len(y_valid) < 10:
            raise ValueError("Not enough non-missing values for CDF matching.")
        xs = np.sort(x_valid)
        ys = np.sort(y_valid)
        # empirical CDF for x
        ranks = np.searchsorted(xs, x, side="right") / max(1, len(xs))
        # inverse CDF in y
        idx = np.clip((ranks * (len(ys) - 1)).astype(int), 0, len(ys) - 1)
        mapped = ys[idx]
        s["__TG_MAPPED__"] = mapped
        thr = qthr(n, nhanes_col)
        s["__LABEL__"] = (s["__TG_MAPPED__"] >= thr).astype(int)
        n["__LABEL__"] = (n[nhanes_col] >= thr).astype(int)
        meta["thr_from_nhanes_quantile"] = thr
        meta["shared_threshold_used"] = thr

    else:
        raise ValueError(f"Unknown label-mode: {mode}")

    meta["synth_label_counts"] = s["__LABEL__"].value_counts(dropna=False).to_dict()
    meta["nhanes_label_counts"] = n["__LABEL__"].value_counts(dropna=False).to_dict()
    return s, n, meta


# -----------------------------
# Modeling
# -----------------------------

def build_preprocessor(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[ColumnTransformer, List[str]]:
    """
    Returns (preprocessor, feature_cols_used).
    """
    cols = [c for c in df.columns if c != target_col and c not in drop_cols]
    # Basic type split: object/category => categorical, else numeric
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, cols


def model_zoo(seed: int) -> Dict[str, BaseEstimator]:
    """
    Models chosen to cover linear + tree ensembles + boosting + MLP.
    Keep them stable and reproducible (fixed random_state).
    """
    return {
        "LR": LogisticRegression(
            max_iter=2000, solver="lbfgs", n_jobs=None, random_state=seed, class_weight=None
        ),
        "RF": RandomForestClassifier(
            n_estimators=600, random_state=seed, n_jobs=-1, max_depth=None,
            min_samples_leaf=2, class_weight=None
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=800, random_state=seed, n_jobs=-1, max_depth=None,
            min_samples_leaf=2, class_weight=None
        ),
        "HGB": HistGradientBoostingClassifier(
            random_state=seed, learning_rate=0.05, max_depth=6, max_iter=600
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=3e-4,
            max_iter=800,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=30,
        ),
    }


@dataclass
class CVResult:
    env: str
    model: str
    fold: int
    metrics: Dict[str, float]


def predict_proba_binary(est: BaseEstimator, X: Any) -> np.ndarray:
    """
    Returns P(y=1).
    """
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        return _sigmoid(est.decision_function(X))
    raise TypeError("Estimator has neither predict_proba nor decision_function.")


def run_cv(
    df: pd.DataFrame,
    env: str,
    target_col: str,
    drop_cols: List[str],
    seed: int,
    n_splits: int,
    n_repeats: int,
    calibrate: bool,
    calibrate_method: str,
    calibrate_cv: int,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Returns:
      - per-fold metrics dataframe
      - per-model aggregated (mean) metrics
    """
    y = df[target_col].astype(int).to_numpy()
    X = df.drop(columns=[target_col])

    pre, _ = build_preprocessor(df, target_col=target_col, drop_cols=drop_cols)

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    zoo = model_zoo(seed)

    fold_rows: List[Dict[str, Any]] = []
    # For consistent fold indexing across models
    splits = list(rkf.split(X, y))

    for model_name, base_model in zoo.items():
        for fold_idx, (tr, te) in enumerate(splits):
            Xtr = X.iloc[tr]
            Xte = X.iloc[te]
            ytr = y[tr]
            yte = y[te]

            est = clone(base_model)

            pipe = Pipeline(steps=[
                ("pre", pre),
                ("clf", est),
            ])

            if calibrate:
                # Calibration must be fit only on training data.
                # CalibratedClassifierCV internally does CV on the training split.
                # We wrap the *full pipeline* (pre + model) so preprocessing is also calibrated safely.
                cal = CalibratedClassifierCV(
                    estimator=pipe,
                    method=calibrate_method,
                    cv=calibrate_cv,
                )
                cal.fit(Xtr, ytr)
                yprob = predict_proba_binary(cal, Xte)
            else:
                pipe.fit(Xtr, ytr)
                yprob = predict_proba_binary(pipe, Xte)

            mets = metric_bundle(yte, yprob)
            row = {"env": env, "model": model_name, "fold": fold_idx, **mets}
            fold_rows.append(row)

    per_fold = pd.DataFrame(fold_rows)
    agg = (
        per_fold
        .groupby(["env", "model"], as_index=False)
        .mean(numeric_only=True)
    )

    per_model: Dict[str, Dict[str, float]] = {}
    for _, r in agg.iterrows():
        per_model[r["model"]] = {k: float(r[k]) for k in ["auroc", "auprc", "logloss", "brier", "ece"]}

    return per_fold, per_model


# -----------------------------
# Report / artifacts
# -----------------------------

def plot_model_flip_involvement(series: pd.Series, outpath: Path, title: str) -> None:
    plt.figure(figsize=(9, 4))
    series = series.reindex(series.index)  # keep order
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.ylabel("Mean flip fraction")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pairwise_heatmap(mat: pd.DataFrame, outpath: Path, title: str) -> None:
    plt.figure(figsize=(7.2, 5.4))
    arr = mat.values
    plt.imshow(arr, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=30, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_summary_txt(outdir: Path, lines: List[str]) -> None:
    (outdir / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_stability_report(
    per_env_model_scores: Dict[str, Dict[str, float]],
    ref_env: str,
    score: str,
    margin: float,
) -> Tuple[RankingStability, pd.DataFrame, pd.Series]:
    """
    per_env_model_scores: env -> model -> score_value
    """
    # Build preference matrices
    envs = list(per_env_model_scores.keys())
    models = sorted(per_env_model_scores[ref_env].keys())

    # Scores for the chosen score metric
    score_by_env: Dict[str, Dict[str, float]] = {}
    for e in envs:
        score_by_env[e] = {m: per_env_model_scores[e].get(m, float("nan")) for m in models}

    pref_ref = preference_matrix(score_by_env[ref_env], margin=margin)

    flip_rates = {}
    spearmans = {}
    consistent = []
    for e in envs:
        pref_e = preference_matrix(score_by_env[e], margin=margin)
        if e == ref_env:
            continue
        fr = flip_fraction(pref_ref, pref_e)
        flip_rates[e] = fr
        # preference consistency = 1 - flip_rate
        consistent.append(1.0 - fr if not np.isnan(fr) else np.nan)

        # Spearman on raw scores
        a = np.array([score_by_env[ref_env][m] for m in models], dtype=float)
        b = np.array([score_by_env[e][m] for m in models], dtype=float)
        spearmans[e] = spearman_corr(a, b)

    global_flip = float(np.nanmean(list(flip_rates.values()))) if flip_rates else float("nan")
    pref_cons = float(np.nanmean(consistent)) if consistent else float("nan")
    spearman_mean = float(np.nanmean(list(spearmans.values()))) if spearmans else float("nan")

    stability = RankingStability(
        ref_env=ref_env,
        score=score,
        margin=margin,
        global_flip_rate=global_flip,
        preference_consistency=pref_cons,
        spearman_mean=spearman_mean,
    )

    # Model-specific involvement: average over other envs
    inv_list = []
    for e in envs:
        if e == ref_env:
            continue
        pref_e = preference_matrix(score_by_env[e], margin=margin)
        inv_list.append(model_flip_involvement(pref_ref, pref_e))
    if inv_list:
        inv = pd.concat(inv_list, axis=1).mean(axis=1).sort_values(ascending=True)
    else:
        inv = pd.Series(dtype=float)

    # Pairwise flip heatmap vs ref: for each pair (i,j), fraction of envs where preference differs
    if len(envs) > 1:
        base = pref_ref.copy()
        frac = np.zeros_like(base.values, dtype=float)
        denom = np.zeros_like(base.values, dtype=float)
        for e in envs:
            if e == ref_env:
                continue
            pref_e = preference_matrix(score_by_env[e], margin=margin)
            A = base.values
            B = pref_e.values
            mask = ~np.isnan(A) & ~np.isnan(B)
            n = A.shape[0]
            for i in range(n):
                mask[i, i] = False
            frac[mask] += (A[mask] != B[mask]).astype(float)
            denom[mask] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.divide(frac, denom)
        pairwise = pd.DataFrame(out, index=base.index, columns=base.columns)
    else:
        pairwise = pref_ref * np.nan

    return stability, pairwise, inv


def selection_policies(
    per_env_metrics: pd.DataFrame,
    rank_score: str,
    ref_env: str,
    flip_rate_by_model: Optional[pd.Series],
    tau: float,
    kappa: float,
) -> pd.DataFrame:
    """
    Returns a table of stability-aware selection outcomes.

    Policies:
      1) select_by_mean_primary: choose highest mean rank_score (across envs)
      2) select_by_worstcase_primary: choose max of worst-case rank_score across envs
      3) select_by_calibration_only: min mean ECE
      4) select_by_rank_stability_only: min flip_rate (if available)
      5) constrained: max mean rank_score s.t. mean ECE<=tau and flip<=kappa
    """
    df = per_env_metrics.copy()
    # Pivot for convenience
    mean_score = df.groupby("model")[rank_score].mean()
    worst_score = df.groupby("model")[rank_score].min()
    mean_ece = df.groupby("model")["ece"].mean()

    rows = []

    def add(policy: str, chosen_model: str, chosen_score: float, ok: bool, extra: Dict[str, Any]) -> None:
        r = {
            "policy": policy,
            "chosen_model": chosen_model,
            "chosen_score": float(chosen_score),
            "constraints_satisfied": bool(ok),
        }
        r.update(extra)
        rows.append(r)

    # 1) mean-primary
    m1 = mean_score.idxmax()
    add("select_by_mean_primary", m1, mean_score[m1], True, {"ece": float(mean_ece[m1])})

    # 2) worstcase-primary
    m2 = worst_score.idxmax()
    add("select_by_worstcase_primary", m2, worst_score[m2], True, {"ece": float(mean_ece[m2])})

    # 3) calibration-only
    m3 = mean_ece.idxmin()
    add("select_by_calibration_only(min ECE)", m3, mean_score.get(m3, float("nan")), True, {"ece": float(mean_ece[m3])})

    # 4) rank-stability only
    if flip_rate_by_model is not None and len(flip_rate_by_model) > 0:
        m4 = flip_rate_by_model.idxmin()
        add("select_by_rank_stability_only(min flip)", m4, mean_score.get(m4, float("nan")), True,
            {"flip_rate": float(flip_rate_by_model[m4]), "ece": float(mean_ece[m4])})
    else:
        add("select_by_rank_stability_only(min flip)", "NA", float("nan"), False, {"flip_rate": float("nan")})

    # 5) constrained
    candidates = mean_score.index.tolist()
    if flip_rate_by_model is None or len(flip_rate_by_model) == 0:
        feasible = [m for m in candidates if (mean_ece[m] <= tau)]
    else:
        feasible = [m for m in candidates if (mean_ece[m] <= tau) and (flip_rate_by_model.get(m, float("inf")) <= kappa)]

    if len(feasible) > 0:
        best = mean_score.loc[feasible].idxmax()
        add("constrained(max score s.t. ece<=tau,flip<=kappa)", best, mean_score[best], True, {
            "tau": tau, "kappa": kappa,
            "ece": float(mean_ece[best]),
            "flip_rate": float(flip_rate_by_model.get(best, float("nan")) if flip_rate_by_model is not None else float("nan")),
        })
    else:
        add("constrained(max score s.t. ece<=tau,flip<=kappa)", "NA", float("nan"), False, {"tau": tau, "kappa": kappa})

    return pd.DataFrame(rows)


# -----------------------------
# Main runners
# -----------------------------

def run_task_mimic(args: argparse.Namespace, outdir: Path) -> None:
    demo = load_csv(args.mimic_demo)
    full = load_csv(args.mimic_full)

    spec = infer_task_spec(args)

    # If target col missing, attempt common alternatives
    for df, name in [(demo, "demo"), (full, "full")]:
        if spec.target_col not in df.columns:
            candidates = [c for c in ["hospital_expire_flag", "mortality", "death", "label"] if c in df.columns]
            if candidates:
                print(f"[warn] target_col '{spec.target_col}' not found in {name}; using '{candidates[0]}'", file=sys.stderr)
                spec.target_col = candidates[0]
            else:
                raise KeyError(f"{name} missing target column '{spec.target_col}'. Provide --target-col.")

    # Drop explicit cols
    drop_cols = list(spec.drop_cols)

    # If id_col is specified, drop it from features but keep it in df
    if spec.id_col and spec.id_col in demo.columns:
        drop_cols.append(spec.id_col)
    if spec.id_col and spec.id_col in full.columns:
        drop_cols.append(spec.id_col)

    env_dfs = {"demo": demo, "full": full}

    per_fold_all = []
    per_env_model = {}
    per_env_metrics_frames = []

    for env, df in env_dfs.items():
        # Ensure binary labels
        yvals = df[spec.target_col].dropna().unique()
        if set(np.unique(yvals)).issubset({0, 1}):
            pass
        else:
            # Try to coerce booleans/strings
            df[spec.target_col] = df[spec.target_col].astype(int)

        per_fold, per_model = run_cv(
            df=df,
            env=env,
            target_col=spec.target_col,
            drop_cols=drop_cols,
            seed=args.seed,
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
            calibrate=args.calibrate,
            calibrate_method=args.calibrate_method,
            calibrate_cv=args.calibrate_cv,
            verbose=True,
        )
        per_fold_all.append(per_fold)
        per_env_model[env] = {m: per_model[m][args.rank_score] for m in per_model.keys()}
        # Store aggregated metrics table for env
        agg = per_fold.groupby(["env", "model"], as_index=False).mean(numeric_only=True)
        per_env_metrics_frames.append(agg)

    per_fold_df = pd.concat(per_fold_all, ignore_index=True)
    per_env_metrics = pd.concat(per_env_metrics_frames, ignore_index=True)

    per_fold_df.to_csv(outdir / "MIMIC_per_fold_metrics.csv", index=False)
    per_env_metrics.to_csv(outdir / "MIMIC_per_env_metrics.csv", index=False)

    stability, pairwise_flip, inv = compute_stability_report(
        per_env_model_scores=per_env_model,
        ref_env=args.ref_env or "demo",
        score=args.rank_score,
        margin=args.rank_margin,
    )

    # Artifacts
    pairwise_flip.to_csv(outdir / "MIMIC_pairwise_flip_heatmap.csv")
    inv.to_csv(outdir / "MIMIC_model_flip_involvement.csv", header=["mean_flip_fraction"])

    plot_model_flip_involvement(inv, outdir / "MIMIC_model_flip_involvement.png",
                                title="Model-specific flip involvement (lower = more stable)")
    plot_pairwise_heatmap(pairwise_flip, outdir / "MIMIC_pairwise_flip_heatmap.png",
                          title=f"Pairwise preference flip fraction vs ref='{stability.ref_env}' (score={stability.score}, margin={stability.margin})")

    # selection policies
    sel = selection_policies(
        per_env_metrics=per_env_metrics,
        rank_score=args.rank_score,
        ref_env=stability.ref_env,
        flip_rate_by_model=inv,
        tau=args.tau,
        kappa=args.kappa,
    )
    sel.to_csv(outdir / "MIMIC_selection_outcomes.csv", index=False)

    report = {
        "task": spec.name,
        "envs": spec.envs,
        "ranking_stability": asdict(stability),
        "args": vars(args),
    }
    (outdir / "MIMIC_ranking_stability_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Summary
    top = (
        per_env_metrics.groupby("model")[args.rank_score].mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={args.rank_score: "mean_"+args.rank_score})
    )
    lines = [
        "=== ICML STABILITY SUITE SUMMARY ===",
        f"Task: {spec.name}",
        f"Environments: {spec.envs}",
        "",
        f"Ranking stability (global): flip_rate={stability.global_flip_rate:.3f}, spearman_mean={stability.spearman_mean:.3f}",
        "",
        f"Top models by mean {args.rank_score}:",
        top.to_string(index=False),
        "",
        "Selection outcomes:",
        sel.to_string(index=False),
    ]
    write_summary_txt(outdir, lines)


def run_task_tg(args: argparse.Namespace, outdir: Path) -> None:
    df_s = load_csv(args.synthetic)
    df_n = load_csv(args.nhanes)

    # label construction
    synth_col = args.synth_tg_col or "TG4h"
    nhanes_col = args.nhanes_tg_col or "TG"

    df_s2, df_n2, meta = build_tg_labels(
        df_synth=df_s,
        df_nhanes=df_n,
        synth_col=synth_col,
        nhanes_col=nhanes_col,
        mode=args.label_mode,
        q=args.label_quantile,
        shared_threshold=args.shared_threshold,
    )
    (outdir / "TG_label_definition.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    spec = infer_task_spec(args)

    drop_cols = list(spec.drop_cols)
    if spec.id_col and spec.id_col in df_s2.columns:
        drop_cols.append(spec.id_col)
    if spec.id_col and spec.id_col in df_n2.columns:
        drop_cols.append(spec.id_col)

    env_dfs = {"synthetic": df_s2, "nhanes": df_n2}

    per_fold_all = []
    per_env_model = {}
    per_env_metrics_frames = []

    for env, df in env_dfs.items():
        per_fold, per_model = run_cv(
            df=df,
            env=env,
            target_col="__LABEL__",
            drop_cols=drop_cols + ["__TG_MAPPED__"] if "__TG_MAPPED__" in df.columns else drop_cols,
            seed=args.seed,
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
            calibrate=args.calibrate,
            calibrate_method=args.calibrate_method,
            calibrate_cv=args.calibrate_cv,
            verbose=True,
        )
        per_fold_all.append(per_fold)
        per_env_model[env] = {m: per_model[m][args.rank_score] for m in per_model.keys()}
        agg = per_fold.groupby(["env", "model"], as_index=False).mean(numeric_only=True)
        per_env_metrics_frames.append(agg)

    per_fold_df = pd.concat(per_fold_all, ignore_index=True)
    per_env_metrics = pd.concat(per_env_metrics_frames, ignore_index=True)

    per_fold_df.to_csv(outdir / "TG_per_fold_metrics.csv", index=False)
    per_env_metrics.to_csv(outdir / "TG_per_env_metrics.csv", index=False)

    stability, pairwise_flip, inv = compute_stability_report(
        per_env_model_scores=per_env_model,
        ref_env=args.ref_env or "synthetic",
        score=args.rank_score,
        margin=args.rank_margin,
    )

    pairwise_flip.to_csv(outdir / "TG_pairwise_flip_heatmap.csv")
    inv.to_csv(outdir / "TG_model_flip_involvement.csv", header=["mean_flip_fraction"])

    plot_model_flip_involvement(inv, outdir / "TG_model_flip_involvement.png",
                                title="Model-specific flip involvement (lower = more stable)")
    plot_pairwise_heatmap(pairwise_flip, outdir / "TG_pairwise_flip_heatmap.png",
                          title=f"Pairwise preference flip fraction vs ref='{stability.ref_env}' (score={stability.score}, margin={stability.margin})")

    sel = selection_policies(
        per_env_metrics=per_env_metrics,
        rank_score=args.rank_score,
        ref_env=stability.ref_env,
        flip_rate_by_model=inv,
        tau=args.tau,
        kappa=args.kappa,
    )
    sel.to_csv(outdir / "TG_selection_outcomes.csv", index=False)

    report = {
        "task": spec.name,
        "envs": spec.envs,
        "ranking_stability": asdict(stability),
        "label_meta": meta,
        "args": vars(args),
    }
    (outdir / "TG_ranking_stability_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    top = (
        per_env_metrics.groupby("model")[args.rank_score].mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={args.rank_score: "mean_"+args.rank_score})
    )
    lines = [
        "=== ICML STABILITY SUITE SUMMARY ===",
        f"Task: {spec.name}",
        f"Environments: {spec.envs}",
        "",
        f"Label meta: {json.dumps(meta, ensure_ascii=False)}",
        "",
        f"Ranking stability (global): flip_rate={stability.global_flip_rate:.3f}, spearman_mean={stability.spearman_mean:.3f}",
        "",
        f"Top models by mean {args.rank_score}:",
        top.to_string(index=False),
        "",
        "Selection outcomes:",
        sel.to_string(index=False),
    ]
    write_summary_txt(outdir, lines)

    # Optional: threshold scan sensitivity (ICML-grade)
    if args.threshold_scan is not None and len(args.threshold_scan) == 3:
        t0, t1, step = args.threshold_scan
        scan_rows = []
        for thr in np.arange(t0, t1 + 1e-9, step):
            try:
                df_s_sc, df_n_sc, meta_sc = build_tg_labels(
                    df_synth=df_s, df_nhanes=df_n,
                    synth_col=synth_col, nhanes_col=nhanes_col,
                    mode="shared_absolute", q=args.label_quantile, shared_threshold=float(thr)
                )
                # quick aggregated AUROC only, fewer repeats for scan
                per_env_model_sc = {}
                for env, dfX in {"synthetic": df_s_sc, "nhanes": df_n_sc}.items():
                    _, per_model_sc = run_cv(
                        df=dfX, env=env, target_col="__LABEL__",
                        drop_cols=drop_cols,
                        seed=args.seed,
                        n_splits=args.cv_splits,
                        n_repeats=max(1, args.cv_repeats // 2),
                        calibrate=args.calibrate,
                        calibrate_method=args.calibrate_method,
                        calibrate_cv=args.calibrate_cv,
                        verbose=False,
                    )
                    per_env_model_sc[env] = {m: per_model_sc[m][args.rank_score] for m in per_model_sc.keys()}
                stab_sc, _, inv_sc = compute_stability_report(
                    per_env_model_scores=per_env_model_sc,
                    ref_env=args.ref_env or "synthetic",
                    score=args.rank_score,
                    margin=args.rank_margin,
                )
                scan_rows.append({
                    "threshold": float(thr),
                    "global_flip_rate": stab_sc.global_flip_rate,
                    "spearman_mean": stab_sc.spearman_mean,
                    "mean_flip_ET": float(inv_sc.get("ET", np.nan)),
                    "mean_flip_HGB": float(inv_sc.get("HGB", np.nan)),
                    "mean_flip_LR": float(inv_sc.get("LR", np.nan)),
                    "mean_flip_RF": float(inv_sc.get("RF", np.nan)),
                    "mean_flip_MLP": float(inv_sc.get("MLP", np.nan)),
                    "synth_pos_rate": float(np.mean(df_s_sc["__LABEL__"] == 1)),
                    "nhanes_pos_rate": float(np.mean(df_n_sc["__LABEL__"] == 1)),
                })
            except Exception:
                continue
        scan_df = pd.DataFrame(scan_rows)
        if len(scan_df) > 0:
            scan_df.to_csv(outdir / "TG_threshold_scan.csv", index=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICML Stability Suite (dataset edition divergence + cross-domain stability)")

    p.add_argument("--task", choices=["mimic_mortality", "tg_high_response"], required=True)

    # Data paths
    p.add_argument("--mimic-demo", dest="mimic_demo", default=None, help="CSV: demo MIMIC analytic dataset")
    p.add_argument("--mimic-full", dest="mimic_full", default=None, help="CSV: full MIMIC analytic dataset")
    p.add_argument("--synthetic", default=None, help="CSV: synthetic dataset")
    p.add_argument("--nhanes", default=None, help="CSV: NHANES dataset")

    # Generic schema controls
    p.add_argument("--target-col", default=None, help="Target column name (for mimic_mortality)")
    p.add_argument("--id-col", default=None, help="Optional ID column to drop from features")
    p.add_argument("--drop-cols", nargs="*", default=[], help="Extra columns to drop from features")

    # TG task schema
    p.add_argument("--synth-tg-col", default=None, help="Synthetic TG column (default TG4h)")
    p.add_argument("--nhanes-tg-col", default=None, help="NHANES TG column (default TG)")

    # TG label harmonization
    p.add_argument("--label-mode", default="nhanes_quantile_shared",
                   choices=["nhanes_quantile_shared", "synth_quantile_shared", "shared_absolute", "env_specific_quantile", "cdf_match"],
                   help="How to define the TG high-response label across environments")
    p.add_argument("--label-quantile", type=float, default=0.75, help="Quantile for label definition when applicable")
    p.add_argument("--shared-threshold", type=float, default=None, help="Absolute threshold for label-mode shared_absolute")
    p.add_argument("--threshold-scan", nargs=3, type=float, default=None,
                   metavar=("T0", "T1", "STEP"),
                   help="Optional: scan shared thresholds for TG task (produces TG_threshold_scan.csv)")

    # CV / evaluation
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=5)

    # Calibration
    p.add_argument("--calibrate", action="store_true", help="Apply probabilistic calibration inside each split")
    p.add_argument("--calibrate-method", default="sigmoid", choices=["sigmoid", "isotonic"])
    p.add_argument("--calibrate-cv", type=int, default=3, help="Internal CV for CalibratedClassifierCV on the training fold")

    # Ranking stability
    p.add_argument("--rank-score", default="auroc", choices=["auroc", "auprc", "logloss", "brier", "ece"])
    p.add_argument("--rank-margin", type=float, default=0.0, help="Preference margin for flips")
    p.add_argument("--ref-env", default=None, help="Reference environment for flips")

    # Stability-aware selection constraints
    p.add_argument("--tau", type=float, default=0.05, help="ECE constraint for constrained selection")
    p.add_argument("--kappa", type=float, default=0.25, help="Flip-rate constraint for constrained selection")

    # Output
    p.add_argument("--outdir", default="runs/icml_suite", help="Output directory")

    args = p.parse_args(argv)

    # basic validation
    if args.task == "mimic_mortality":
        if not args.mimic_demo or not args.mimic_full:
            p.error("--mimic-demo and --mimic-full are required for task mimic_mortality")
    if args.task == "tg_high_response":
        if not args.synthetic or not args.nhanes:
            p.error("--synthetic and --nhanes are required for task tg_high_response")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)

    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    # Save args for reproducibility
    (outdir / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if args.task == "mimic_mortality":
        run_task_mimic(args, outdir)
    elif args.task == "tg_high_response":
        run_task_tg(args, outdir)
    else:
        raise ValueError(args.task)

    print(f"[done] artifacts written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
