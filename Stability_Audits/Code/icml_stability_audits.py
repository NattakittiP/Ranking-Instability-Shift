#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICML Stability Next Audits
=========================
This script extends your current ICML stability pipeline with three *publication-grade* audits:

1) Leakage & Trivial-Predictor Audit
2) Controlled Sample-Size & Cohort-Composition Controls (MIMIC demo vs full)
3) Mechanism / "Why it flips" diagnostics

It is designed to be *model-agnostic* and *schema-robust*:
- Works even if feature columns differ slightly between demo/full.
- Avoids target leakage by fitting preprocessing inside each split.
- Produces interpretable artifacts (CSV/JSON/plots) that you can drop into an ICML paper.

Dependencies: numpy, pandas, scikit-learn, matplotlib

Examples
--------
# 1) Leakage audit on MIMIC (demo and full)
python icml_stability_next_audits.py leakage_audit \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --outdir runs/audits_leakage_mimic

# 2) MIMIC controls: sample-size + composition
python icml_stability_next_audits.py mimic_controls \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --controls sample_size composition ipw \
  --n-reps 50 \
  --outdir runs/audits_controls_mimic

# 3) Mechanism: why rankings flip (uses both MIMIC and TG cross-domain if provided)
python icml_stability_next_audits.py flip_mechanisms \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --synthetic Synthetic_Dataset_1500_Patients_precise.csv \
  --nhanes nhanes_rsce_dataset_clean.csv \
  --tg-synth-col TG4h --tg-nhanes-col TG \
  --outdir runs/audits_mechanisms
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# --------------------------
# Reproducibility utilities
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

def infer_binary_target(df: pd.DataFrame, target: str) -> pd.Series:
    y = df[target]
    if y.dtype == bool:
        return y.astype(int)
    if y.dtype == object:
        return y.astype(str).str.strip().replace({"False": "0", "True": "1"}).astype(int)
    return y.astype(int)

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


# --------------------------
# Core modeling primitives
# --------------------------

def base_models(seed: int) -> Dict[str, BaseEstimator]:
    return {
        "LR": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=600, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "HGB": HistGradientBoostingClassifier(random_state=seed, learning_rate=0.05, max_depth=6, max_iter=600),
    }

def cv_predict(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    model: BaseEstimator,
    seed: int,
    n_splits: int,
    n_repeats: int,
    tqdm_desc: str = "CV",
) -> Tuple[np.ndarray, np.ndarray]:
    y = infer_binary_target(df, target).to_numpy()
    X = df.drop(columns=[target])

    pre, _, _, _ = build_preprocessor(df, target, drop_cols)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    yprob = np.full(shape=len(df), fill_value=np.nan, dtype=float)

    total_folds = n_splits * n_repeats
    with tqdm(total=total_folds, desc=tqdm_desc, dynamic_ncols=True) as pbar:
        for tr, te in rkf.split(X, y):
            pipe = Pipeline(steps=[("pre", pre), ("clf", clone(model))])
            pipe.fit(X.iloc[tr], y[tr])
            yprob[te] = predict_proba_1(pipe, X.iloc[te])
            pbar.update(1)

    return y, yprob


# ============================================================
# (1) Leakage & Trivial-Predictor Audit
# ============================================================

LEAKY_NAME_HINTS = [
    "target", "label", "outcome", "mort", "death", "expire", "survival",
    "discharge", "los", "length_of_stay", "icu_los", "hospital_los",
    "readmit", "readmission",
    "time_to_death", "dod", "date_of_death",
    "after", "post", "followup",
    "vent", "vasopressor", "dialysis",
]

def leaky_name_score(col: str) -> float:
    c = col.lower()
    return float(sum(1 for h in LEAKY_NAME_HINTS if h in c))

def single_feature_screen(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    seed: int,
    n_splits: int = 5,
    tqdm_desc: str = "Single-feature screen",
) -> pd.DataFrame:
    y = infer_binary_target(df, target).to_numpy()
    cand = []
    for c in df.columns:
        if c == target or c in drop_cols:
            continue
        if df[c].dtype == object:
            continue
        cand.append(c)

    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in tqdm(cand, desc=tqdm_desc, dynamic_ncols=True):
        x = df[[col]].copy()
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ])
        yprob = np.full(len(df), np.nan)

        with tqdm(total=n_splits, desc=f"  CV folds: {col}", dynamic_ncols=True, leave=False) as pbar:
            for tr, te in skf.split(x, y):
                pipe.fit(x.iloc[tr], y[tr])
                yprob[te] = pipe.predict_proba(x.iloc[te])[:, 1]
                pbar.update(1)

        mets = metric_bundle(y, yprob)
        rows.append({
            "feature": col,
            "single_feature_auroc": mets["auroc"],
            "single_feature_auprc": mets["auprc"],
            "missing_rate": float(np.mean(df[col].isna())),
            "leaky_name_score": leaky_name_score(col),
        })

    return pd.DataFrame(rows).sort_values(["single_feature_auroc", "single_feature_auprc"], ascending=False)

def permutation_leakage_test(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    seed: int,
    model_name: str = "HGB",
    n_splits: int = 5,
    n_repeats: int = 3,
    n_perm_repeats: int = 10,
    tqdm_desc: str = "Permutation leakage test",
) -> pd.DataFrame:
    models = base_models(seed)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models)}")

    y = infer_binary_target(df, target).to_numpy()
    X = df.drop(columns=[target])

    pre, used_cols, _, _ = build_preprocessor(df, target, drop_cols)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    importances = []
    total_folds = n_splits * n_repeats

    with tqdm(total=total_folds, desc=tqdm_desc, dynamic_ncols=True) as fold_bar:
        for fold_i, (tr, te) in enumerate(rkf.split(X, y), start=1):
            pipe = Pipeline(steps=[("pre", pre), ("clf", clone(models[model_name]))])
            pipe.fit(X.iloc[tr], y[tr])
            base_prob = predict_proba_1(pipe, X.iloc[te])
            base = roc_auc_score(y[te], base_prob) if len(np.unique(y[te])) > 1 else np.nan

            with tqdm(total=len(used_cols), desc=f"  Fold {fold_i}/{total_folds}: permute features", dynamic_ncols=True, leave=False) as feat_bar:
                for col in used_cols:
                    drops = []
                    with tqdm(total=n_perm_repeats, desc=f"    {col}: perm repeats", dynamic_ncols=True, leave=False) as rep_bar:
                        for r in range(n_perm_repeats):
                            Xp = X.iloc[te].copy()
                            rng = np.random.default_rng(seed + 1000 + r)
                            Xp[col] = rng.permutation(Xp[col].to_numpy())
                            pp = predict_proba_1(pipe, Xp)
                            sc = roc_auc_score(y[te], pp) if len(np.unique(y[te])) > 1 else np.nan
                            drops.append(base - sc)
                            rep_bar.update(1)

                    importances.append({
                        "feature": col,
                        "auc_drop_mean": float(np.nanmean(drops)),
                        "auc_drop_std": float(np.nanstd(drops)),
                        "leaky_name_score": leaky_name_score(col),
                    })
                    feat_bar.update(1)

            fold_bar.update(1)

    imp = (
        pd.DataFrame(importances)
        .groupby("feature", as_index=False)
        .agg({"auc_drop_mean": "mean", "auc_drop_std": "mean", "leaky_name_score": "mean"})
        .sort_values("auc_drop_mean", ascending=False)
    )
    return imp

def run_leakage_audit(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    demo = load_csv(args.demo, max_rows=args.max_rows, seed=args.seed)
    full = load_csv(args.full, max_rows=args.max_rows, seed=args.seed)

    for df, name in [(demo, "demo"), (full, "full")]:
        if args.target not in df.columns:
            raise KeyError(f"[{name}] missing target '{args.target}'. Available head: {list(df.columns)[:30]}")

    drop_cols = list(args.drop_cols or [])
    if args.id_col:
        drop_cols.append(args.id_col)

    common = sorted(list((set(demo.columns) & set(full.columns)) - set([args.target]) - set(drop_cols)))
    demo2 = demo[[*common, args.target]].copy()
    full2 = full[[*common, args.target]].copy()

    # Name hint scores
    tqdm.write("[leakage_audit] computing leaky name hints...")
    name_scores = pd.DataFrame({
        "feature": common,
        "leaky_name_score": [leaky_name_score(c) for c in tqdm(common, desc="Name hint scan", dynamic_ncols=True)],
    }).sort_values("leaky_name_score", ascending=False)
    name_scores.to_csv(outdir / "leakage_name_hints.csv", index=False)

    # Single feature screen
    tqdm.write("[leakage_audit] single-feature screens...")
    sf_demo = single_feature_screen(demo2, args.target, drop_cols=[], seed=args.seed, n_splits=args.cv_splits, tqdm_desc="Demo: single-feature screen")
    sf_full = single_feature_screen(full2, args.target, drop_cols=[], seed=args.seed, n_splits=args.cv_splits, tqdm_desc="Full: single-feature screen")
    sf_demo.to_csv(outdir / "demo_single_feature_screen.csv", index=False)
    sf_full.to_csv(outdir / "full_single_feature_screen.csv", index=False)

    # Permutation leakage test
    tqdm.write("[leakage_audit] permutation leakage tests...")
    imp_demo = permutation_leakage_test(
        demo2, args.target, drop_cols=[],
        seed=args.seed, model_name=args.model, n_splits=args.cv_splits, n_repeats=args.cv_repeats,
        n_perm_repeats=args.n_perm_repeats,
        tqdm_desc="Demo: permutation leakage",
    )
    imp_full = permutation_leakage_test(
        full2, args.target, drop_cols=[],
        seed=args.seed, model_name=args.model, n_splits=args.cv_splits, n_repeats=args.cv_repeats,
        n_perm_repeats=args.n_perm_repeats,
        tqdm_desc="Full: permutation leakage",
    )
    imp_demo.to_csv(outdir / "demo_permutation_auc_drop.csv", index=False)
    imp_full.to_csv(outdir / "full_permutation_auc_drop.csv", index=False)

    def plot_top(df: pd.DataFrame, title: str, path: Path) -> None:
        top = df.dropna().head(20).iloc[::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(top["feature"].astype(str), top["single_feature_auroc"].astype(float))
        plt.title(title)
        plt.xlabel("Single-feature AUROC (CV)")
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    tqdm.write("[leakage_audit] plotting...")
    plot_top(sf_demo, "Demo: top single-feature AUROC (trivial predictor screen)", outdir / "demo_top_single_feature_auroc.png")
    plot_top(sf_full, "Full: top single-feature AUROC (trivial predictor screen)", outdir / "full_top_single_feature_auroc.png")

    def summarize(sf: pd.DataFrame, imp: pd.DataFrame) -> Dict[str, Any]:
        sf2 = sf.dropna()
        return {
            "n_features_screened": int(len(sf2)),
            "max_single_feature_auroc": float(sf2["single_feature_auroc"].max()) if len(sf2) else None,
            "n_features_auroc_ge_0.95": int(np.sum(sf2["single_feature_auroc"] >= 0.95)) if len(sf2) else 0,
            "top_suspects_by_single_feature_auroc": sf2.head(10)[["feature", "single_feature_auroc", "leaky_name_score"]].to_dict(orient="records"),
            "top_suspects_by_permutation_auc_drop": imp.head(10)[["feature", "auc_drop_mean", "leaky_name_score"]].to_dict(orient="records"),
        }

    summary = {
        "target": args.target,
        "common_feature_count": int(len(common)),
        "demo_summary": summarize(sf_demo, imp_demo),
        "full_summary": summarize(sf_full, imp_full),
        "notes": [
            "High single-feature AUROC and large permutation AUC-drop are red flags for leakage/proxies.",
            "If suspects contain discharge/death/LOS or post-event interventions, consider excluding them or reframing the prediction time horizon.",
        ],
    }
    save_json(outdir / "leakage_audit_summary.json", summary)
    print(f"[done] leakage audit artifacts written to: {outdir.resolve()}")


# ============================================================
# (2) Controlled Sample-Size & Cohort-Composition Controls (MIMIC)
# ============================================================

def pick_anchor_covariates(df_demo: pd.DataFrame, df_full: pd.DataFrame, target: str, max_k: int = 12) -> List[str]:
    common = list((set(df_demo.columns) & set(df_full.columns)) - {target})
    cand = []
    for c in tqdm(common, desc="Pick anchor covariates", dynamic_ncols=True):
        if df_demo[c].dtype == object or df_full[c].dtype == object:
            continue
        mr = max(float(df_demo[c].isna().mean()), float(df_full[c].isna().mean()))
        if mr > 0.35:
            continue
        v1 = float(np.nanvar(df_demo[c].to_numpy()))
        v2 = float(np.nanvar(df_full[c].to_numpy()))
        if (v1 <= 1e-10) or (v2 <= 1e-10):
            continue
        cand.append((mr, -(v1+v2), c))
    cand = sorted(cand)
    return [c for _, _, c in cand[:max_k]]

def propensity_scores(
    df_demo: pd.DataFrame,
    df_full: pd.DataFrame,
    covariates: List[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    A = df_demo[covariates].copy()
    B = df_full[covariates].copy()
    A["__env__"] = 1
    B["__env__"] = 0
    Z = pd.concat([A, B], ignore_index=True)

    y = Z["__env__"].to_numpy()
    X = Z.drop(columns=["__env__"])

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    pA = p[:len(A)]
    pB = p[len(A):]
    return pA, pB

def ipw_weights_for_full(p_demo_full: np.ndarray, clip: Tuple[float, float] = (0.05, 0.95)) -> np.ndarray:
    p = np.clip(p_demo_full, clip[0], clip[1])
    w = p / (1.0 - p)
    w = w / np.mean(w)
    return w

def weighted_bootstrap_sample(
    df: pd.DataFrame,
    n: int,
    weights: Optional[np.ndarray],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if weights is None:
        idx = rng.choice(len(df), size=n, replace=False if n <= len(df) else True)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)
        idx = rng.choice(len(df), size=n, replace=True, p=w)
    return df.iloc[idx].reset_index(drop=True)

def run_controls_one_rep(
    df_demo: pd.DataFrame,
    df_full: pd.DataFrame,
    target: str,
    seed: int,
    model: BaseEstimator,
    drop_cols: List[str],
    n_splits: int,
    n_repeats: int,
    control_mode: str,
    covariates: List[str],
) -> Dict[str, Any]:
    n_demo = len(df_demo)

    if control_mode == "sample_size":
        full_ctrl = df_full.sample(n=n_demo, random_state=seed).reset_index(drop=True)
    elif control_mode in ("composition", "ipw"):
        _, p_full = propensity_scores(df_demo, df_full, covariates=covariates, seed=seed)
        w_full = ipw_weights_for_full(p_full)
        full_ctrl = weighted_bootstrap_sample(df_full, n=n_demo, weights=w_full, seed=seed)
    else:
        raise ValueError(control_mode)

    y_d, p_d = cv_predict(
        df_demo, target, drop_cols=drop_cols, model=model, seed=seed,
        n_splits=n_splits, n_repeats=n_repeats,
        tqdm_desc=f"CV demo ({control_mode}, seed={seed})",
    )
    y_f, p_f = cv_predict(
        full_ctrl, target, drop_cols=drop_cols, model=model, seed=seed,
        n_splits=n_splits, n_repeats=n_repeats,
        tqdm_desc=f"CV full_ctrl ({control_mode}, seed={seed})",
    )

    mets_demo = metric_bundle(y_d, p_d)
    mets_full = metric_bundle(y_f, p_f)

    return {
        "rep_seed": seed,
        "control_mode": control_mode,
        "demo": mets_demo,
        "full_controlled": mets_full,
        "delta_auroc_full_minus_demo": float(mets_full["auroc"] - mets_demo["auroc"]),
        "delta_logloss_full_minus_demo": float(mets_full["logloss"] - mets_demo["logloss"]),
    }

def run_mimic_controls(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    demo = load_csv(args.demo, max_rows=None, seed=args.seed)
    full = load_csv(args.full, max_rows=None, seed=args.seed)

    for df, name in [(demo, "demo"), (full, "full")]:
        if args.target not in df.columns:
            raise KeyError(f"[{name}] missing target '{args.target}'.")

    drop_cols = list(args.drop_cols or [])
    if args.id_col:
        drop_cols.append(args.id_col)

    common = sorted(list((set(demo.columns) & set(full.columns)) - set([args.target]) - set(drop_cols)))
    demo2 = demo[[*common, args.target]].copy()
    full2 = full[[*common, args.target]].copy()

    cov = args.covariates or pick_anchor_covariates(demo2, full2, args.target, max_k=args.max_covariates)
    save_json(outdir / "chosen_anchor_covariates.json", {"covariates": cov})

    model = base_models(args.seed)[args.model]

    rows = []
    total_jobs = args.n_reps * len(args.controls)
    with tqdm(total=total_jobs, desc="MIMIC controls: reps × modes", dynamic_ncols=True) as job_bar:
        for r in range(args.n_reps):
            rep_seed = args.seed + 100*r + 7
            for mode in args.controls:
                tqdm.write(f"[mimic_controls] rep={r+1}/{args.n_reps}, mode={mode}, seed={rep_seed}")
                rows.append(run_controls_one_rep(
                    df_demo=demo2, df_full=full2, target=args.target, seed=rep_seed,
                    model=model, drop_cols=[], n_splits=args.cv_splits, n_repeats=args.cv_repeats,
                    control_mode=mode, covariates=cov,
                ))
                job_bar.update(1)

    df_out = pd.DataFrame([{
        "rep_seed": o["rep_seed"],
        "control_mode": o["control_mode"],
        "model": args.model,
        "demo_auroc": o["demo"]["auroc"],
        "demo_auprc": o["demo"]["auprc"],
        "demo_logloss": o["demo"]["logloss"],
        "full_auroc": o["full_controlled"]["auroc"],
        "full_auprc": o["full_controlled"]["auprc"],
        "full_logloss": o["full_controlled"]["logloss"],
        "delta_auroc_full_minus_demo": o["delta_auroc_full_minus_demo"],
        "delta_logloss_full_minus_demo": o["delta_logloss_full_minus_demo"],
    } for o in rows])

    df_out.to_csv(outdir / "mimic_controls_replication_table.csv", index=False)

    summ = (
        df_out.groupby("control_mode", as_index=False)
        .agg({"delta_auroc_full_minus_demo": ["mean", "std"], "demo_auroc": ["mean", "std"], "full_auroc": ["mean", "std"]})
    )
    summ.columns = ["_".join([c for c in col if c]) for col in summ.columns.to_flat_index()]
    summ.to_csv(outdir / "mimic_controls_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    order = df_out["control_mode"].unique().tolist()
    data = [df_out.loc[df_out["control_mode"] == m, "delta_auroc_full_minus_demo"].dropna().to_numpy() for m in order]
    plt.boxplot(data, labels=order, showmeans=True)
    plt.axhline(0.0, linestyle="--")
    plt.title("MIMIC controls: ΔAUROC distribution (full_controlled - demo)")
    plt.ylabel("Δ AUROC")
    plt.tight_layout()
    plt.savefig(outdir / "mimic_controls_delta_auroc_boxplot.png", dpi=220)
    plt.close()

    print(f"[done] mimic controls artifacts written to: {outdir.resolve()}")


# ============================================================
# (3) Mechanism / "Why it flips"
# ============================================================

def population_stability_index(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    a = a.astype(float)
    b = b.astype(float)
    x = np.concatenate([a[~np.isnan(a)], b[~np.isnan(b)]])
    if len(x) < 20:
        return float("nan")
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")

    def hist(v: np.ndarray) -> np.ndarray:
        vv = v[~np.isnan(v)]
        h, _ = np.histogram(vv, bins=edges)
        p = h / max(1, np.sum(h))
        return np.clip(p, 1e-6, 1.0)

    pa = hist(a)
    pb = hist(b)
    m = min(len(pa), len(pb))
    pa, pb = pa[:m], pb[:m]
    return float(np.sum((pa - pb) * np.log(pa / pb)))

def domain_shift_report(df_A: pd.DataFrame, df_B: pd.DataFrame, target: str, drop_cols: List[str], max_features: int = 60) -> pd.DataFrame:
    common = sorted(list((set(df_A.columns) & set(df_B.columns)) - {target} - set(drop_cols)))
    rows = []
    for c in tqdm(common, desc="Domain shift report (per feature)", dynamic_ncols=True):
        mrA = float(df_A[c].isna().mean())
        mrB = float(df_B[c].isna().mean())
        psi = float("nan") if (df_A[c].dtype == object or df_B[c].dtype == object) else population_stability_index(df_A[c].to_numpy(), df_B[c].to_numpy())
        rows.append({"feature": c, "missing_rate_A": mrA, "missing_rate_B": mrB, "missing_delta_B_minus_A": mrB - mrA, "psi_numeric": psi, "leaky_name_score": leaky_name_score(c)})
    out = pd.DataFrame(rows)
    out["psi_abs"] = out["psi_numeric"].abs()
    return out.sort_values(["psi_abs", "missing_delta_B_minus_A"], ascending=False).head(max_features)

def top_feature_ablation_flip(
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    target: str,
    seed: int,
    model: BaseEstimator,
    drop_cols: List[str],
    candidate_features: List[str],
    n_splits: int,
    n_repeats: int,
) -> pd.DataFrame:
    yA, pA = cv_predict(df_A, target, drop_cols=drop_cols, model=model, seed=seed, n_splits=n_splits, n_repeats=n_repeats, tqdm_desc="Ablation baseline: CV A")
    yB, pB = cv_predict(df_B, target, drop_cols=drop_cols, model=model, seed=seed, n_splits=n_splits, n_repeats=n_repeats, tqdm_desc="Ablation baseline: CV B")
    baseA = metric_bundle(yA, pA)
    baseB = metric_bundle(yB, pB)

    rows = [{"ablation": "baseline_none", "k": 0, "auroc_A": baseA["auroc"], "auroc_B": baseB["auroc"], "delta_auroc_B_minus_A": baseB["auroc"] - baseA["auroc"]}]

    k_list = [1, 3, 5, 8, 12, 20]
    for k in tqdm(k_list, desc="Ablation over k", dynamic_ncols=True):
        feats = candidate_features[:k]
        yA2, pA2 = cv_predict(df_A, target, drop_cols=drop_cols + feats, model=model, seed=seed, n_splits=n_splits, n_repeats=n_repeats, tqdm_desc=f"Ablation drop k={k}: CV A")
        yB2, pB2 = cv_predict(df_B, target, drop_cols=drop_cols + feats, model=model, seed=seed, n_splits=n_splits, n_repeats=n_repeats, tqdm_desc=f"Ablation drop k={k}: CV B")
        mA = metric_bundle(yA2, pA2)
        mB = metric_bundle(yB2, pB2)
        rows.append({"ablation": "drop_top_shift_features", "k": k, "auroc_A": mA["auroc"], "auroc_B": mB["auroc"], "delta_auroc_B_minus_A": mB["auroc"] - mA["auroc"]})
    return pd.DataFrame(rows)

def run_flip_mechanisms(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    demo = load_csv(args.demo, seed=args.seed)
    full = load_csv(args.full, seed=args.seed)

    for df, name in [(demo, "demo"), (full, "full")]:
        if args.target not in df.columns:
            raise KeyError(f"[{name}] missing target '{args.target}'.")

    drop_cols = list(args.drop_cols or [])
    if args.id_col:
        drop_cols.append(args.id_col)

    common = sorted(list((set(demo.columns) & set(full.columns)) - {args.target} - set(drop_cols)))
    demo2 = demo[[*common, args.target]].copy()
    full2 = full[[*common, args.target]].copy()

    tqdm.write("[flip_mechanisms] computing domain shift report...")
    shift = domain_shift_report(demo2, full2, target=args.target, drop_cols=[], max_features=args.max_shift_features)
    shift.to_csv(outdir / "mimic_top_shift_features.csv", index=False)

    tqdm.write("[flip_mechanisms] running ablation study...")
    model = base_models(args.seed)[args.model]
    cand_features = shift["feature"].tolist()
    ablation = top_feature_ablation_flip(
        demo2, full2, target=args.target, seed=args.seed, model=model,
        drop_cols=[], candidate_features=cand_features,
        n_splits=args.cv_splits, n_repeats=args.cv_repeats
    )
    ablation.to_csv(outdir / "mimic_shift_feature_ablation.csv", index=False)

    tqdm.write("[flip_mechanisms] plotting...")
    plt.figure(figsize=(8, 4.8))
    plt.plot(ablation["k"], ablation["delta_auroc_B_minus_A"], marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.title("Mechanism probe (MIMIC): dropping top shift features vs ΔAUROC (full - demo)")
    plt.xlabel("k top-shift features dropped")
    plt.ylabel("Δ AUROC")
    plt.tight_layout()
    plt.savefig(outdir / "mimic_ablation_delta_auroc_curve.png", dpi=220)
    plt.close()

    print(f"[done] mechanism artifacts written to: {outdir.resolve()}")


# --------------------------
# CLI
# --------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICML Stability Next Audits (Leakage, Controls, Mechanisms)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("leakage_audit", help="Leakage & trivial-predictor audit (demo vs full)")
    pa.add_argument("--demo", required=True)
    pa.add_argument("--full", required=True)
    pa.add_argument("--target", required=True)
    pa.add_argument("--id-col", default=None)
    pa.add_argument("--drop-cols", nargs="*", default=[])
    pa.add_argument("--model", default="HGB", choices=["LR", "RF", "HGB"])
    pa.add_argument("--cv-splits", type=int, default=5)
    pa.add_argument("--cv-repeats", type=int, default=3)
    pa.add_argument("--n-perm-repeats", type=int, default=10)
    pa.add_argument("--max-rows", type=int, default=None)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--outdir", required=True)

    pc = sub.add_parser("mimic_controls", help="Sample-size & composition controls for MIMIC edition divergence")
    pc.add_argument("--demo", required=True)
    pc.add_argument("--full", required=True)
    pc.add_argument("--target", required=True)
    pc.add_argument("--id-col", default=None)
    pc.add_argument("--drop-cols", nargs="*", default=[])
    pc.add_argument("--model", default="HGB", choices=["LR", "RF", "HGB"])
    pc.add_argument("--controls", nargs="+", default=["sample_size", "composition", "ipw"], choices=["sample_size", "composition", "ipw"])
    pc.add_argument("--covariates", nargs="*", default=None)
    pc.add_argument("--max-covariates", type=int, default=12)
    pc.add_argument("--n-reps", type=int, default=50)
    pc.add_argument("--cv-splits", type=int, default=5)
    pc.add_argument("--cv-repeats", type=int, default=3)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--outdir", required=True)

    pm = sub.add_parser("flip_mechanisms", help="Mechanism probes: shifts + ablation diagnostics")
    pm.add_argument("--demo", required=True)
    pm.add_argument("--full", required=True)
    pm.add_argument("--target", required=True)
    pm.add_argument("--id-col", default=None)
    pm.add_argument("--drop-cols", nargs="*", default=[])
    pm.add_argument("--model", default="HGB", choices=["LR", "RF", "HGB"])
    pm.add_argument("--cv-splits", type=int, default=5)
    pm.add_argument("--cv-repeats", type=int, default=3)
    pm.add_argument("--max-shift-features", type=int, default=60)
    pm.add_argument("--seed", type=int, default=42)
    pm.add_argument("--outdir", required=True)

    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    outdir = Path(args.outdir)
    safe_makedirs(outdir)
    save_json(outdir / "run_args.json", vars(args))

    if args.cmd == "leakage_audit":
        run_leakage_audit(args)
    elif args.cmd == "mimic_controls":
        run_mimic_controls(args)
    elif args.cmd == "flip_mechanisms":
        run_flip_mechanisms(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()

