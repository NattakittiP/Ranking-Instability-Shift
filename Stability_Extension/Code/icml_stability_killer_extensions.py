#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICML Stability Killer Extensions
================================
This script adds three "ICML-grade" extensions on top of your existing suite:

(1) Patient-level & time-aware split audits
    - GroupKFold/GroupShuffleSplit by subject_id (or any group column)
    - Time split (train early, test late) if you provide a time column

(2) Composition controls with diagnostics + learning curves (MIMIC demo vs full)
    - Propensity score overlap plots (positivity)
    - Covariate balance before/after weighting (SMD)
    - Learning curves: divergence vs sample size n

(3) Mechanism decomposition (Why it flips)
    - C2ST (edition classifier): can we tell demo vs full? (domain separability)
    - Label-shift diagnostics: prevalence + black-box shift (BBSD) check
    - Missingness-transfer test: impose demo missingness on full and re-evaluate

Outputs are paper-ready CSV/JSON/PNG.

Dependencies
-----------
numpy, pandas, scikit-learn, matplotlib

Usage
-----
# (1) Group & time split audits
python icml_stability_killer_extensions.py split_audit \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --group-col subject_id \
  --time-col admittime \
  --outdir runs/killer_split_audit

# (2) Controls + diagnostics + learning curves
python icml_stability_killer_extensions.py controls_plus \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --group-col subject_id \
  --n-grid 500 1000 2000 4000 8000 \
  --n-reps 30 \
  --outdir runs/killer_controls_plus

# (3) Mechanism decomposition
python icml_stability_killer_extensions.py mechanisms_plus \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --group-col subject_id \
  --outdir runs/killer_mechanisms_plus

Notes
-----
- If you don't have subject_id/time columns, you can omit them, but ICML reviewers usually expect them.
- This code is conservative: all preprocessing is fit inside each training split (no leakage).
"""

from __future__ import annotations

import argparse
import json
import os
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_csv(path: str, max_rows: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df

def infer_binary_target(df: pd.DataFrame, target: str) -> np.ndarray:
    y = df[target]
    if y.dtype == bool:
        return y.astype(int).to_numpy()
    if y.dtype == object:
        return y.astype(str).str.strip().replace({"False": "0", "True": "1"}).astype(int).to_numpy()
    return y.astype(int).to_numpy()

def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "logloss": np.nan, "brier": np.nan}
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, np.c_[1-y_prob, y_prob], labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

def predict_proba_1(est: BaseEstimator, X: Any) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))
    raise TypeError("Estimator lacks predict_proba and decision_function")

def as_datetime_safe(s: pd.Series) -> pd.Series:
    # Robust parsing across common MIMIC formats
    return pd.to_datetime(s, errors="coerce", utc=True)

# --------------------------
# Preprocessing
# --------------------------

def build_preprocessor(df: pd.DataFrame, target: str, drop_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    cols = [c for c in df.columns if c != target and c not in drop_cols]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, cols, num_cols, cat_cols

def base_models(seed: int) -> Dict[str, BaseEstimator]:
    return {
        "LR": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=700, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "HGB": HistGradientBoostingClassifier(random_state=seed, learning_rate=0.05, max_depth=6, max_iter=700),
    }

# ============================================================
# (1) Split-policy audits
# ============================================================

def groupkfold_oof_probs(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    model: BaseEstimator,
    seed: int,
    n_splits: int = 5,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    drop_cols = list(drop_cols or [])
    if group_col not in drop_cols:
        drop_cols.append(group_col)

    y = infer_binary_target(df, target)
    groups = df[group_col].to_numpy()
    X = df.drop(columns=[target])

    pre, _, _, _ = build_preprocessor(df, target, drop_cols=drop_cols)
    gkf = GroupKFold(n_splits=n_splits)

    yprob = np.full(len(df), np.nan, dtype=float)
    for tr, te in gkf.split(X, y, groups=groups):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        yprob[te] = predict_proba_1(pipe, X.iloc[te])

    return y, yprob

def group_shuffle_holdout(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    model: BaseEstimator,
    seed: int,
    test_size: float = 0.2,
    n_reps: int = 20,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    drop_cols = list(drop_cols or [])
    if group_col not in drop_cols:
        drop_cols.append(group_col)

    y = infer_binary_target(df, target)
    groups = df[group_col].to_numpy()
    X = df.drop(columns=[target])

    pre, _, _, _ = build_preprocessor(df, target, drop_cols=drop_cols)
    gss = GroupShuffleSplit(n_splits=n_reps, test_size=test_size, random_state=seed)

    rows = []
    for i, (tr, te) in enumerate(gss.split(X, y, groups=groups), start=1):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        p = predict_proba_1(pipe, X.iloc[te])
        m = metric_bundle(y[te], p)
        rows.append({"rep": i, **m, "n_train": int(len(tr)), "n_test": int(len(te))})
    return pd.DataFrame(rows)

def time_split_holdout(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    model: BaseEstimator,
    seed: int,
    train_frac: float = 0.7,
    drop_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    drop_cols = list(drop_cols or [])
    if time_col not in drop_cols:
        drop_cols.append(time_col)

    dt = as_datetime_safe(df[time_col])
    ok = dt.notna()
    d2 = df.loc[ok].copy()
    dt2 = dt.loc[ok].sort_values()

    # split by time quantile
    split_t = dt2.quantile(train_frac)
    tr_idx = d2.index[as_datetime_safe(d2[time_col]) <= split_t].to_numpy()
    te_idx = d2.index[as_datetime_safe(d2[time_col]) > split_t].to_numpy()

    y = infer_binary_target(d2, target)
    X = d2.drop(columns=[target])

    pre, _, _, _ = build_preprocessor(d2, target, drop_cols=drop_cols)
    pipe = Pipeline([("pre", pre), ("clf", clone(model))])
    pipe.fit(X.loc[tr_idx], y[np.isin(d2.index.to_numpy(), tr_idx)])
    p = predict_proba_1(pipe, X.loc[te_idx])

    y_te = y[np.isin(d2.index.to_numpy(), te_idx)]
    m = metric_bundle(y_te, p)
    return {
        "train_frac": train_frac,
        "split_time_utc": str(split_t),
        "n_total_timeparseable": int(len(d2)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "metrics": m,
    }

def run_split_audit(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir); safe_makedirs(outdir)
    demo = load_csv(args.demo, seed=args.seed)
    full = load_csv(args.full, seed=args.seed)

    drop_cols = list(args.drop_cols or [])
    if args.id_cols:
        drop_cols.extend(args.id_cols)

    common = sorted(list((set(demo.columns) & set(full.columns)) - {args.target} - set(drop_cols)))
    demo2 = demo[[*common, args.target] + ([args.group_col] if args.group_col and args.group_col in demo.columns and args.group_col not in common else [])].copy()
    full2 = full[[*common, args.target] + ([args.group_col] if args.group_col and args.group_col in full.columns and args.group_col not in common else [])].copy()
    if args.time_col:
        if args.time_col in demo.columns and args.time_col not in demo2.columns:
            demo2[args.time_col] = demo[args.time_col]
        if args.time_col in full.columns and args.time_col not in full2.columns:
            full2[args.time_col] = full[args.time_col]

    model = base_models(args.seed)[args.model]

    results = {"target": args.target, "model": args.model, "common_feature_count": int(len(common))}

    # Baseline (stratified)
    def stratified_oof(df: pd.DataFrame) -> Dict[str, Any]:
        y = infer_binary_target(df, args.target)
        X = df.drop(columns=[args.target])
        pre, _, _, _ = build_preprocessor(df, args.target, drop_cols=[])
        skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
        yprob = np.full(len(df), np.nan)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([("pre", pre), ("clf", clone(model))])
            pipe.fit(X.iloc[tr], y[tr])
            yprob[te] = predict_proba_1(pipe, X.iloc[te])
        return {"metrics": metric_bundle(y, yprob), "n": int(len(df))}

    results["demo_stratified_cv"] = stratified_oof(demo2)
    results["full_stratified_cv"] = stratified_oof(full2)

    # Group audits
    if args.group_col and args.group_col in demo2.columns and args.group_col in full2.columns:
        yd, pd_ = groupkfold_oof_probs(demo2, args.target, args.group_col, model, args.seed, n_splits=args.cv_splits)
        yf, pf_ = groupkfold_oof_probs(full2, args.target, args.group_col, model, args.seed, n_splits=args.cv_splits)
        results["demo_groupkfold_cv"] = {"metrics": metric_bundle(yd, pd_), "n": int(len(demo2)), "group_col": args.group_col}
        results["full_groupkfold_cv"] = {"metrics": metric_bundle(yf, pf_), "n": int(len(full2)), "group_col": args.group_col}

        gsd = group_shuffle_holdout(demo2, args.target, args.group_col, model, args.seed, n_reps=args.n_reps, test_size=args.test_size)
        gsf = group_shuffle_holdout(full2, args.target, args.group_col, model, args.seed, n_reps=args.n_reps, test_size=args.test_size)
        gsd.to_csv(outdir / "demo_group_shuffle_holdout.csv", index=False)
        gsf.to_csv(outdir / "full_group_shuffle_holdout.csv", index=False)

        # Plot distribution
        plt.figure(figsize=(8.5, 4.8))
        plt.boxplot([gsd["auroc"].dropna(), gsf["auroc"].dropna()], labels=["demo", "full"], showmeans=True)
        plt.title("GroupShuffleSplit AUROC distribution")
        plt.ylabel("AUROC")
        plt.tight_layout()
        plt.savefig(outdir / "group_shuffle_auroc_boxplot.png", dpi=220)
        plt.close()

    # Time audit
    if args.time_col and args.time_col in demo2.columns and args.time_col in full2.columns:
        results["demo_time_split"] = time_split_holdout(demo2, args.target, args.time_col, model, args.seed, train_frac=args.train_frac)
        results["full_time_split"] = time_split_holdout(full2, args.target, args.time_col, model, args.seed, train_frac=args.train_frac)

    save_json(outdir / "split_audit_summary.json", results)
    print(f"[done] split audit written to {outdir.resolve()}")

# ============================================================
# (2) Controls+ diagnostics + learning curves
# ============================================================

def pick_numeric_covariates(df_demo: pd.DataFrame, df_full: pd.DataFrame, target: str, max_k: int = 15) -> List[str]:
    common = list((set(df_demo.columns) & set(df_full.columns)) - {target})
    cands = []
    for c in common:
        if df_demo[c].dtype == object or df_full[c].dtype == object:
            continue
        mr = max(df_demo[c].isna().mean(), df_full[c].isna().mean())
        if mr > 0.35:
            continue
        v = np.nanvar(df_demo[c].to_numpy()) + np.nanvar(df_full[c].to_numpy())
        if v <= 1e-10:
            continue
        cands.append((mr, -v, c))
    cands = sorted(cands)
    return [c for _, _, c in cands[:max_k]]

def fit_propensity(df_demo: pd.DataFrame, df_full: pd.DataFrame, covariates: List[str], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    A = df_demo[covariates].copy(); A["__env__"] = 1
    B = df_full[covariates].copy(); B["__env__"] = 0
    Z = pd.concat([A, B], ignore_index=True)
    y = Z["__env__"].to_numpy()
    X = Z.drop(columns=["__env__"])

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, random_state=seed)),
    ])
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]  # P(demo|x)
    return p[:len(A)], p[len(A):]

def ipw_for_full(p_demo_full: np.ndarray, clip: Tuple[float,float]=(0.05,0.95)) -> np.ndarray:
    p = np.clip(p_demo_full, clip[0], clip[1])
    w = p/(1.0-p)
    w = w/np.mean(w)
    return w

def smd(a: np.ndarray, b: np.ndarray, wa: Optional[np.ndarray]=None, wb: Optional[np.ndarray]=None) -> float:
    a = a.astype(float); b = b.astype(float)
    maska = ~np.isnan(a); maskb = ~np.isnan(b)
    a = a[maska]; b = b[maskb]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    if wa is None:
        ma = np.mean(a); va = np.var(a, ddof=1)
    else:
        wa = wa[maska]; wa = wa/np.sum(wa)
        ma = np.sum(wa*a)
        va = np.sum(wa*(a-ma)**2)
    if wb is None:
        mb = np.mean(b); vb = np.var(b, ddof=1)
    else:
        wb = wb[maskb]; wb = wb/np.sum(wb)
        mb = np.sum(wb*b)
        vb = np.sum(wb*(b-mb)**2)
    sp = math.sqrt((va+vb)/2.0) if (va+vb) > 0 else np.nan
    return float((mb-ma)/sp) if sp and not np.isnan(sp) else float("nan")

def balance_table(df_demo: pd.DataFrame, df_full: pd.DataFrame, covariates: List[str], w_full: Optional[np.ndarray]=None) -> pd.DataFrame:
    rows = []
    for c in covariates:
        a = df_demo[c].to_numpy()
        b = df_full[c].to_numpy()
        rows.append({
            "covariate": c,
            "missing_demo": float(np.mean(pd.isna(a))),
            "missing_full": float(np.mean(pd.isna(b))),
            "smd_unweighted": smd(a, b, wa=None, wb=None),
            "smd_weighted_full": smd(a, b, wa=None, wb=w_full) if w_full is not None else np.nan
        })
    return pd.DataFrame(rows).sort_values("smd_unweighted", ascending=False)

def stratified_cv_oof(df: pd.DataFrame, target: str, model: BaseEstimator, seed: int, n_splits: int=5) -> Dict[str, float]:
    y = infer_binary_target(df, target)
    X = df.drop(columns=[target])
    pre, _, _, _ = build_preprocessor(df, target, drop_cols=[])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = np.full(len(df), np.nan)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = predict_proba_1(pipe, X.iloc[te])
    return metric_bundle(y, prob)

def run_controls_plus(args: argparse.Namespace) -> None:
    import math
    outdir = Path(args.outdir); safe_makedirs(outdir)
    demo = load_csv(args.demo, seed=args.seed)
    full = load_csv(args.full, seed=args.seed)

    # Restrict to common columns (and keep group/time out of features)
    drop_cols = list(args.drop_cols or [])
    if args.group_col:
        drop_cols.append(args.group_col)
    if args.time_col:
        drop_cols.append(args.time_col)

    common = sorted(list((set(demo.columns) & set(full.columns)) - {args.target} - set(drop_cols)))
    demo2 = demo[[*common, args.target]].copy()
    full2 = full[[*common, args.target]].copy()

    model = base_models(args.seed)[args.model]

    # Propensity + diagnostics
    cov = args.covariates or pick_numeric_covariates(demo2, full2, args.target, max_k=args.max_covariates)
    p_demo, p_full = fit_propensity(demo2, full2, cov, seed=args.seed)
    w_full = ipw_for_full(p_full)

    # overlap plot
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(p_demo, bins=30, alpha=0.7, label="demo: P(demo|x)")
    plt.hist(p_full, bins=30, alpha=0.7, label="full: P(demo|x)")
    plt.title("Propensity overlap (positivity check)")
    plt.xlabel("P(demo | x)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "propensity_overlap.png", dpi=220)
    plt.close()

    bt = balance_table(demo2, full2, covariates=cov, w_full=w_full)
    bt.to_csv(outdir / "balance_table_smd.csv", index=False)
    save_json(outdir / "propensity_meta.json", {"covariates": cov, "w_full_mean": float(np.mean(w_full)), "w_full_max": float(np.max(w_full)), "w_full_p99": float(np.quantile(w_full, 0.99))})

    # Learning curves: n-grid
    rows = []
    n_grid = list(args.n_grid)
    n_demo_total = len(demo2)
    rng = np.random.default_rng(args.seed)

    for n in n_grid:
        for r in range(args.n_reps):
            seed_r = args.seed + 1000*r + 17
            # sample demo without replacement
            n_eff = min(n, n_demo_total)
            demo_s = demo2.sample(n=n_eff, random_state=seed_r).reset_index(drop=True)
            # sample full to same size (unweighted) and IPW-weighted bootstrap
            full_unw = full2.sample(n=n_eff, random_state=seed_r).reset_index(drop=True)

            # IPW bootstrap to match composition
            w = w_full / np.sum(w_full)
            idx = rng.choice(len(full2), size=n_eff, replace=True, p=w)
            full_ipw = full2.iloc[idx].reset_index(drop=True)

            m_demo = stratified_cv_oof(demo_s, args.target, model, seed=seed_r, n_splits=args.cv_splits)
            m_full_unw = stratified_cv_oof(full_unw, args.target, model, seed=seed_r, n_splits=args.cv_splits)
            m_full_ipw = stratified_cv_oof(full_ipw, args.target, model, seed=seed_r, n_splits=args.cv_splits)

            rows.append({
                "n": int(n_eff), "rep": int(r+1),
                "demo_auroc": m_demo["auroc"],
                "full_unw_auroc": m_full_unw["auroc"],
                "full_ipw_auroc": m_full_ipw["auroc"],
                "delta_unw": m_full_unw["auroc"] - m_demo["auroc"],
                "delta_ipw": m_full_ipw["auroc"] - m_demo["auroc"],
            })

    df_lc = pd.DataFrame(rows)
    df_lc.to_csv(outdir / "learning_curve_table.csv", index=False)

    # Plot mean delta vs n
    summ = df_lc.groupby("n", as_index=False).agg({"delta_unw":["mean","std"], "delta_ipw":["mean","std"]})
    summ.columns = ["_".join([c for c in col if c]) for col in summ.columns.to_flat_index()]
    summ.to_csv(outdir / "learning_curve_summary.csv", index=False)

    plt.figure(figsize=(8.5, 4.8))
    plt.errorbar(summ["n"], summ["delta_unw_mean"], yerr=summ["delta_unw_std"], marker="o", label="ΔAUROC full(unw) - demo")
    plt.errorbar(summ["n"], summ["delta_ipw_mean"], yerr=summ["delta_ipw_std"], marker="o", label="ΔAUROC full(IPW) - demo")

    plt.axhline(0.0, linestyle="--")
    plt.title("Learning curves: edition divergence vs sample size")
    plt.xlabel("n (matched)")
    plt.ylabel("Δ AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve_delta_auroc.png", dpi=220)
    plt.close()

    save_json(outdir / "controls_plus_meta.json", {
        "target": args.target,
        "model": args.model,
        "common_feature_count": int(len(common)),
        "n_grid": n_grid,
        "n_reps": args.n_reps,
        "cv_splits": args.cv_splits,
        "interpretation": [
            "Propensity overlap + balance (SMD) are required to claim composition control is valid.",
            "Learning curve shows whether divergence is a finite-sample artifact or persists asymptotically.",
        ],
    })
    print(f"[done] controls_plus written to {outdir.resolve()}")

# ============================================================
# (3) Mechanism decomposition (C2ST, label shift, missingness transfer)
# ============================================================

def c2st_edition_classifier(
    df_demo: pd.DataFrame,
    df_full: pd.DataFrame,
    seed: int,
    model_name: str = "HGB",
    test_frac: float = 0.25,
) -> Dict[str, Any]:
    # Build binary task: env=1 for demo, 0 for full
    A = df_demo.copy(); A["__env__"] = 1
    B = df_full.copy(); B["__env__"] = 0
    Z = pd.concat([A, B], ignore_index=True)
    y = Z["__env__"].to_numpy()
    X = Z.drop(columns=["__env__"])

    pre, _, _, _ = build_preprocessor(Z, target="__env__", drop_cols=[])
    clf = base_models(seed)[model_name]

    # stratified split
    skf = StratifiedKFold(n_splits=int(1/test_frac), shuffle=True, random_state=seed)
    tr, te = next(iter(skf.split(X, y)))
    pipe = Pipeline([("pre", pre), ("clf", clone(clf))])
    pipe.fit(X.iloc[tr], y[tr])
    p = predict_proba_1(pipe, X.iloc[te])
    auroc = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else float("nan")
    return {
        "edition_classifier": model_name,
        "test_frac": test_frac,
        "n_total": int(len(Z)),
        "auroc": float(auroc),
    }

def missingness_transfer(full: pd.DataFrame, demo: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[str,float]]:
    # Impose demo marginal missingness rates onto full (columnwise)
    rng = np.random.default_rng(seed)
    out = full.copy()
    rates = {}
    for c in out.columns:
        r = float(demo[c].isna().mean())
        rates[c] = r
        if r <= 0:
            continue
        mask = rng.random(len(out)) < r
        out.loc[mask, c] = np.nan
    return out, rates

def bb_shift_score(y_source_prob: np.ndarray, y_target_prob: np.ndarray, n_bins: int = 20) -> float:
    # Simple distribution divergence of predicted probabilities (proxy for label shift / concept shift)
    a = y_source_prob.astype(float)
    b = y_target_prob.astype(float)
    edges = np.linspace(0, 1, n_bins+1)
    ha, _ = np.histogram(a, bins=edges); hb, _ = np.histogram(b, bins=edges)
    pa = (ha + 1e-6) / np.sum(ha + 1e-6)
    pb = (hb + 1e-6) / np.sum(hb + 1e-6)
    return float(np.sum((pa - pb) * np.log(pa / pb)))  # PSI-like

def oof_probs_stratified(df: pd.DataFrame, target: str, model: BaseEstimator, seed: int, n_splits: int=5) -> Tuple[np.ndarray, np.ndarray]:
    y = infer_binary_target(df, target)
    X = df.drop(columns=[target])
    pre, _, _, _ = build_preprocessor(df, target, drop_cols=[])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = np.full(len(df), np.nan)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = predict_proba_1(pipe, X.iloc[te])
    return y, prob

def train_calibrated_source_and_eval_target(
    source: pd.DataFrame,
    target_df: pd.DataFrame,
    target: str,
    base_model: BaseEstimator,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Fit on source with internal CV calibration; evaluate on target
    y_s = infer_binary_target(source, target)
    X_s = source.drop(columns=[target])
    y_t = infer_binary_target(target_df, target)
    X_t = target_df.drop(columns=[target])

    pre, _, _, _ = build_preprocessor(source, target, drop_cols=[])
    base_pipe = Pipeline([("pre", pre), ("clf", clone(base_model))])
    # Calibrate using 3-fold CV on the source
    cal = CalibratedClassifierCV(base_pipe, method="isotonic", cv=3)
    cal.fit(X_s, y_s)
    p_s = cal.predict_proba(X_s)[:, 1]
    p_t = cal.predict_proba(X_t)[:, 1]
    return p_s, p_t

def run_mechanisms_plus(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir); safe_makedirs(outdir)

    demo_raw = load_csv(args.demo, seed=args.seed)
    full_raw = load_csv(args.full, seed=args.seed)

    drop_cols = list(args.drop_cols or [])
    if args.group_col:
        drop_cols.append(args.group_col)
    if args.time_col:
        drop_cols.append(args.time_col)

    common = sorted(list((set(demo_raw.columns) & set(full_raw.columns)) - {args.target} - set(drop_cols)))
    demo = demo_raw[[*common, args.target]].copy()
    full = full_raw[[*common, args.target]].copy()

    model = base_models(args.seed)[args.model]

    # 3.1 C2ST (edition classifier)
    c2 = c2st_edition_classifier(demo.drop(columns=[args.target]), full.drop(columns=[args.target]), seed=args.seed, model_name=args.model)
    save_json(outdir / "c2st_edition_classifier.json", c2)

    # 3.2 Label shift / black-box shift using calibrated source->target predicted probabilities
    p_demo_on_demo, p_demo_on_full = train_calibrated_source_and_eval_target(
        source=demo, target_df=full, target=args.target, base_model=model, seed=args.seed
    )
    shift = {
        "demo_prevalence": float(np.mean(infer_binary_target(demo, args.target) == 1)),
        "full_prevalence": float(np.mean(infer_binary_target(full, args.target) == 1)),
        "bb_shift_score_PSI_like": bb_shift_score(p_demo_on_demo, p_demo_on_full),
        "notes": [
            "Large prevalence gap and large probability-distribution divergence suggest label shift and/or concept shift.",
            "This is a diagnostic; it doesn't 'solve' the shift—use it to argue mechanisms and limits of demo-as-proxy.",
        ],
    }
    save_json(outdir / "label_shift_diagnostics.json", shift)

    plt.figure(figsize=(8.5, 4.8))
    plt.hist(p_demo_on_demo, bins=30, alpha=0.7, label="calibrated model scores on demo")
    plt.hist(p_demo_on_full, bins=30, alpha=0.7, label="same model scores on full")
    plt.title("Black-box shift diagnostic (score distribution)")
    plt.xlabel("Predicted probability")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bb_shift_score_hist.png", dpi=220)
    plt.close()

    # 3.3 Missingness transfer test (impose demo missingness on full)
    full_miss, rates = missingness_transfer(full.drop(columns=[args.target]), demo.drop(columns=[args.target]), seed=args.seed)
    full_miss2 = full_miss.copy()
    full_miss2[args.target] = full[args.target].to_numpy()

    # Evaluate baseline OOF on demo and full, then on full-with-demo-missingness
    yD, pD = oof_probs_stratified(demo, args.target, model, seed=args.seed, n_splits=args.cv_splits)
    yF, pF = oof_probs_stratified(full, args.target, model, seed=args.seed, n_splits=args.cv_splits)
    yFm, pFm = oof_probs_stratified(full_miss2, args.target, model, seed=args.seed, n_splits=args.cv_splits)

    mD = metric_bundle(yD, pD); mF = metric_bundle(yF, pF); mFm = metric_bundle(yFm, pFm)
    miss_test = {
        "metrics_demo": mD,
        "metrics_full": mF,
        "metrics_full_with_demo_missingness": mFm,
        "delta_full_minus_demo": float(mF["auroc"] - mD["auroc"]),
        "delta_fullMissing_minus_demo": float(mFm["auroc"] - mD["auroc"]),
        "delta_fullMissing_minus_full": float(mFm["auroc"] - mF["auroc"]),
        "interpretation": [
            "If full_with_demo_missingness moves closer to demo, missingness regime explains part of edition divergence.",
            "If it moves away (or barely changes), mechanism is not missingness marginals alone—likely label/cohort/extraction differences.",
        ],
    }
    save_json(outdir / "missingness_transfer_test.json", miss_test)
    save_json(outdir / "demo_missingness_rates_used.json", {"rates": rates})

    # Small plot
    plt.figure(figsize=(7.8, 4.6))
    xs = ["demo", "full", "full(+demo-missing)"]
    ys = [mD["auroc"], mF["auroc"], mFm["auroc"]]
    plt.bar(xs, ys)
    plt.title("Missingness-transfer probe (AUROC)")
    plt.ylabel("AUROC (stratified CV OOF)")
    plt.tight_layout()
    plt.savefig(outdir / "missingness_transfer_auroc_bar.png", dpi=220)
    plt.close()

    # Save meta
    save_json(outdir / "mechanisms_plus_meta.json", {
        "target": args.target,
        "model": args.model,
        "common_feature_count": int(len(common)),
        "cv_splits": args.cv_splits,
        "key_claims_supported": [
            "C2ST AUROC quantifies edition separability (domain gap).",
            "Label prevalence + BBSD score support label/concept shift as a mechanism.",
            "Missingness-transfer isolates missingness regime contribution.",
        ],
    })

    print(f"[done] mechanisms_plus written to {outdir.resolve()}")

# --------------------------
# CLI
# --------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICML Stability Killer Extensions (split audits, controls+, mechanisms+)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("split_audit", help="Patient-level & time-aware split audits")
    s1.add_argument("--demo", required=True)
    s1.add_argument("--full", required=True)
    s1.add_argument("--target", required=True)
    s1.add_argument("--group-col", default=None, help="e.g., subject_id")
    s1.add_argument("--time-col", default=None, help="e.g., admittime / charttime")
    s1.add_argument("--train-frac", type=float, default=0.7)
    s1.add_argument("--test-size", type=float, default=0.2)
    s1.add_argument("--n-reps", type=int, default=20)
    s1.add_argument("--cv-splits", type=int, default=5)
    s1.add_argument("--model", default="HGB", choices=["LR","RF","HGB"])
    s1.add_argument("--drop-cols", nargs="*", default=[])
    s1.add_argument("--id-cols", nargs="*", default=[])
    s1.add_argument("--seed", type=int, default=42)
    s1.add_argument("--outdir", required=True)

    s2 = sub.add_parser("controls_plus", help="Controls + diagnostics + learning curves")
    s2.add_argument("--demo", required=True)
    s2.add_argument("--full", required=True)
    s2.add_argument("--target", required=True)
    s2.add_argument("--group-col", default=None)
    s2.add_argument("--time-col", default=None)
    s2.add_argument("--covariates", nargs="*", default=None)
    s2.add_argument("--max-covariates", type=int, default=15)
    s2.add_argument("--n-grid", nargs="+", type=int, default=[500,1000,2000,4000,8000])
    s2.add_argument("--n-reps", type=int, default=30)
    s2.add_argument("--cv-splits", type=int, default=5)
    s2.add_argument("--model", default="HGB", choices=["LR","RF","HGB"])
    s2.add_argument("--drop-cols", nargs="*", default=[])
    s2.add_argument("--seed", type=int, default=42)
    s2.add_argument("--outdir", required=True)

    s3 = sub.add_parser("mechanisms_plus", help="Mechanism decomposition (C2ST, label shift, missingness transfer)")
    s3.add_argument("--demo", required=True)
    s3.add_argument("--full", required=True)
    s3.add_argument("--target", required=True)
    s3.add_argument("--group-col", default=None)
    s3.add_argument("--time-col", default=None)
    s3.add_argument("--cv-splits", type=int, default=5)
    s3.add_argument("--model", default="HGB", choices=["LR","RF","HGB"])
    s3.add_argument("--drop-cols", nargs="*", default=[])
    s3.add_argument("--seed", type=int, default=42)
    s3.add_argument("--outdir", required=True)

    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    outdir = Path(args.outdir); safe_makedirs(outdir)
    save_json(outdir / "run_args.json", vars(args))

    if args.cmd == "split_audit":
        run_split_audit(args)
    elif args.cmd == "controls_plus":
        run_controls_plus(args)
    elif args.cmd == "mechanisms_plus":
        run_mechanisms_plus(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()
