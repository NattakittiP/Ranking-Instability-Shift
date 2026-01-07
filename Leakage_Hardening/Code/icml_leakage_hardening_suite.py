#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICML Leakage Hardening & Near-Perfect-Performance Sanity Suite
==============================================================
Use this when you observe suspiciously high performance (e.g., AUROC ~ 0.99+)
and want to *prove* to reviewers that results are not due to trivial leakage.

What it runs (paper-grade):
(1) Label permutation sanity check (AUROC/AUPRC should collapse to ~0.5 / prevalence)
(2) Feature-block ablations:
    - demographics only
    - admission meta only
    - labs only
    - demographics + admission meta (no labs)
    - admission meta + labs (no demographics)
(3) Missingness & measurement-leak simulation:
    - "early-only" proxy: randomly keep only X% of labs (simulate early availability)
(4) Hardness sweep:
    - LR vs HGB vs RF under the same CV (optionally GroupShuffleSplit by subject_id)

Outputs: CSV/JSON/PNG.

Run
---
python icml_leakage_hardening_suite.py run \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --group-col subject_id \
  --outdir runs/leakage_hardening
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

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_y(df: pd.DataFrame, target: str) -> np.ndarray:
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

def proba1(est: BaseEstimator, X: Any) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
        z = np.clip(z, -40, 40)
        return 1.0/(1.0+np.exp(-z))
    raise TypeError("Estimator lacks predict_proba and decision_function")

def base_models(seed: int) -> Dict[str, BaseEstimator]:
    return {
        "LR": LogisticRegression(max_iter=4000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=800, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "HGB": HistGradientBoostingClassifier(random_state=seed, learning_rate=0.05, max_depth=6, max_iter=800),
    }

def build_preprocessor(df: pd.DataFrame, target: str, drop_cols: List[str]) -> ColumnTransformer:
    cols = [c for c in df.columns if c != target and c not in drop_cols]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                             remainder="drop", verbose_feature_names_out=False)

def stratified_oof(df: pd.DataFrame, target: str, model: BaseEstimator, seed: int,
                   n_splits: int = 5, drop_cols: Optional[List[str]] = None) -> Dict[str, float]:
    drop_cols = list(drop_cols or [])
    y = infer_y(df, target)
    X = df.drop(columns=[target])
    pre = build_preprocessor(df, target, drop_cols=drop_cols)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = np.full(len(df), np.nan)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = proba1(pipe, X.iloc[te])
    return metric_bundle(y, p)

def group_holdout(df: pd.DataFrame, target: str, group_col: str, model: BaseEstimator,
                  seed: int, test_size: float = 0.2, drop_cols: Optional[List[str]] = None) -> Dict[str, float]:
    drop_cols = list(drop_cols or [])
    if group_col not in drop_cols and group_col in df.columns:
        drop_cols.append(group_col)
    y = infer_y(df, target)
    X = df.drop(columns=[target])
    groups = df[group_col].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(iter(gss.split(X, y, groups=groups)))
    pre = build_preprocessor(df, target, drop_cols=drop_cols)
    pipe = Pipeline([("pre", pre), ("clf", clone(model))])
    pipe.fit(X.iloc[tr], y[tr])
    p = proba1(pipe, X.iloc[te])
    return metric_bundle(y[te], p)

def get_blocks(cols: List[str]) -> Dict[str, List[str]]:
    demog = [c for c in cols if c in {"gender", "anchor_age", "race", "marital_status"}]
    admit = [c for c in cols if c in {"insurance", "admission_type", "admission_location"}]
    labs = [c for c in cols if c.startswith("lab_")]
    return {
        "demographics_only": demog,
        "admission_meta_only": admit,
        "labs_only": labs,
        "demographics_plus_admit": demog + admit,
        "admit_plus_labs": admit + labs,
        "all_features": demog + admit + labs,
    }

def simulate_early_labs(df: pd.DataFrame, keep_frac: float, seed: int) -> pd.DataFrame:
    """Randomly drop (set NaN) a fraction of lab columns to simulate early availability."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    lab_cols = [c for c in out.columns if c.startswith("lab_")]
    if not lab_cols:
        return out
    # For each row, keep only keep_frac of lab columns
    k = max(1, int(round(len(lab_cols) * keep_frac)))
    for i in range(len(out)):
        keep = rng.choice(lab_cols, size=k, replace=False)
        drop = [c for c in lab_cols if c not in set(keep)]
        out.loc[i, drop] = np.nan
    return out

def run_suite(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir); safe_makedirs(outdir)
    demo = load_csv(args.demo)
    full = load_csv(args.full)

    # Common space (include group col only for splitting, never as feature)
    id_like = {"subject_id", "hadm_id", "stay_id", "icustay_id"}
    common = sorted(list((set(demo.columns) & set(full.columns)) - {args.target}))
    cols = [c for c in common if c not in id_like]
    demo2 = demo[[*cols, args.target] + ([args.group_col] if args.group_col in demo.columns else [])].copy()
    full2 = full[[*cols, args.target] + ([args.group_col] if args.group_col in full.columns else [])].copy()

    blocks = get_blocks(cols)

    rows = []
    perm_rows = []

    for model_name, model in base_models(args.seed).items():
        # Baseline (all features) stratified CV
        rows.append({"dataset": "demo", "model": model_name, "setting": "stratified_cv_all", **stratified_oof(demo2[[*cols, args.target]], args.target, model, args.seed, args.cv_splits)})
        rows.append({"dataset": "full", "model": model_name, "setting": "stratified_cv_all", **stratified_oof(full2[[*cols, args.target]], args.target, model, args.seed, args.cv_splits)})

        # Group holdout if available
        if args.group_col and args.group_col in demo2.columns and args.group_col in full2.columns:
            rows.append({"dataset": "demo", "model": model_name, "setting": "group_holdout_all", **group_holdout(demo2[[*cols, args.target, args.group_col]], args.target, args.group_col, model, args.seed, test_size=args.test_size)})
            rows.append({"dataset": "full", "model": model_name, "setting": "group_holdout_all", **group_holdout(full2[[*cols, args.target, args.group_col]], args.target, args.group_col, model, args.seed, test_size=args.test_size)})

        # Block ablations
        for block_name, block_cols in blocks.items():
            if not block_cols:
                continue
            dA = demo2[[*block_cols, args.target] + ([args.group_col] if args.group_col in demo2.columns else [])].copy()
            dB = full2[[*block_cols, args.target] + ([args.group_col] if args.group_col in full2.columns else [])].copy()
            rows.append({"dataset": "demo", "model": model_name, "setting": f"stratified_cv_{block_name}", **stratified_oof(dA[[*block_cols, args.target]], args.target, model, args.seed, args.cv_splits)})
            rows.append({"dataset": "full", "model": model_name, "setting": f"stratified_cv_{block_name}", **stratified_oof(dB[[*block_cols, args.target]], args.target, model, args.seed, args.cv_splits)})

        # Label permutation sanity (repeat)
        for r in range(args.perm_repeats):
            rs = args.seed + 1013*r + (7 if model_name == "LR" else 19 if model_name=="RF" else 31)
            y_demo = infer_y(demo2, args.target)
            y_full = infer_y(full2, args.target)
            rng = np.random.default_rng(rs)
            y_demo_perm = rng.permutation(y_demo)
            y_full_perm = rng.permutation(y_full)

            # Evaluate quickly with stratified CV OOF but with permuted labels
            def perm_eval(dfX: pd.DataFrame, y_perm: np.ndarray) -> Dict[str, float]:
                X = dfX.drop(columns=[args.target])
                pre = build_preprocessor(dfX, args.target, drop_cols=[])
                skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=rs)
                p = np.full(len(dfX), np.nan)
                for tr, te in skf.split(X, y_perm):
                    pipe = Pipeline([("pre", pre), ("clf", clone(model))])
                    pipe.fit(X.iloc[tr], y_perm[tr])
                    p[te] = proba1(pipe, X.iloc[te])
                return metric_bundle(y_perm, p)

            m1 = perm_eval(demo2[[*cols, args.target]], y_demo_perm)
            m2 = perm_eval(full2[[*cols, args.target]], y_full_perm)
            perm_rows.append({"dataset": "demo", "model": model_name, "rep": r+1, **m1})
            perm_rows.append({"dataset": "full", "model": model_name, "rep": r+1, **m2})

        # Early-labs simulation (only meaningful if labs exist)
        for keep_frac in args.early_keep_fracs:
            dA = simulate_early_labs(demo2[[*cols, args.target]].copy(), keep_frac=float(keep_frac), seed=args.seed+77)
            dB = simulate_early_labs(full2[[*cols, args.target]].copy(), keep_frac=float(keep_frac), seed=args.seed+99)
            rows.append({"dataset": "demo", "model": model_name, "setting": f"early_labs_keep_{keep_frac}", **stratified_oof(dA, args.target, model, args.seed, args.cv_splits)})
            rows.append({"dataset": "full", "model": model_name, "setting": f"early_labs_keep_{keep_frac}", **stratified_oof(dB, args.target, model, args.seed, args.cv_splits)})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "leakage_hardening_results.csv", index=False)

    perm = pd.DataFrame(perm_rows)
    perm.to_csv(outdir / "label_permutation_sanity.csv", index=False)

    # Plot permutation AUROC distribution
    plt.figure(figsize=(8.5, 4.8))
    for ds in ["demo", "full"]:
        vals = perm.loc[perm["dataset"] == ds, "auroc"].dropna().to_numpy()
        plt.hist(vals, bins=25, alpha=0.7, label=f"{ds} perm AUROC")
    plt.axvline(0.5, linestyle="--")
    plt.title("Label permutation sanity (AUROC should ~0.5)")
    plt.xlabel("AUROC")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "label_permutation_auroc_hist.png", dpi=220)
    plt.close()

    # Summaries
    summary = {
        "target": args.target,
        "cv_splits": args.cv_splits,
        "perm_repeats": args.perm_repeats,
        "early_keep_fracs": args.early_keep_fracs,
        "recommendations": [
            "If full AUROC remains ~0.99+ using ONLY demographics/admission meta, there is likely post-outcome leakage.",
            "If near-perfect AUROC requires labs and collapses under early-labs simulation, argue measurement timing as a mechanism.",
            "Permutation AUROC must be near 0.5; otherwise pipeline or split has leakage.",
        ],
    }
    save_json(outdir / "leakage_hardening_summary.json", summary)
    print(f"[done] leakage hardening written to {outdir.resolve()}")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICML Leakage Hardening Suite")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--demo", required=True)
    r.add_argument("--full", required=True)
    r.add_argument("--target", required=True)
    r.add_argument("--group-col", default="subject_id")
    r.add_argument("--cv-splits", type=int, default=5)
    r.add_argument("--test-size", type=float, default=0.2)
    r.add_argument("--perm-repeats", type=int, default=10)
    r.add_argument("--early-keep-fracs", nargs="*", default=["0.25", "0.5", "0.75"])
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--outdir", required=True)
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    safe_makedirs(Path(args.outdir))
    if args.cmd == "run":
        run_suite(args)

if __name__ == "__main__":
    main()
