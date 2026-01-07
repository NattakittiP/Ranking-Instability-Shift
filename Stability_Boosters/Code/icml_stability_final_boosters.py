#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICML Stability Final Boosters
=============================
You already ran:
- split_audit (group-based),
- controls_plus (composition + overlap/balance),
- mechanisms_plus (C2ST, label-shift, missingness-transfer).

This script adds the remaining "reviewer-killer" experiments that typically
separate a strong ICML paper from a good one:

(4) Time-aware validity (temporal split) + year-wise drift
(5) Proper learning curves for small demo (n-grid auto) + uncertainty
(6) Cross-edition transfer: train on demo -> test full, and train on full -> test demo
(7) Bootstrap CIs (group-aware) for AUROC gaps + stability statistics
(8) Subgroup audits (gender/race/age bins) to show robustness/limits

Outputs: paper-ready CSV/JSON/PNG in one outdir.

Dependencies: numpy, pandas, scikit-learn, matplotlib

Run
---
python icml_stability_final_boosters.py run \
  --demo demo_analytic_dataset_mortality_all_admissions.csv \
  --full full_analytic_dataset_mortality_all_admissions.csv \
  --target label_mortality \
  --group-col subject_id \
  --time-col anchor_year \
  --outdir runs/final_boosters

Notes
-----
- If demo lacks subject_id/time columns, the script will auto-skip those parts and
  write a warning into summary.json. For ICML, you ideally re-export demo to include
  subject_id and a time column (anchor_year / admittime).
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

def _make_columns_unique(cols: List[str]) -> List[str]:
    """
    Make column names unique by appending __dup{n} suffix when duplicates appear.
    Preserves order, keeps all columns (no dropping).
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # strip first (common source of "hidden duplicates" like "age " vs "age")
    df.columns = [str(c).strip() for c in df.columns]
    # enforce unique column names (prevents sklearn ColumnTransformer crashes)
    if df.columns.duplicated().any():
        df.columns = _make_columns_unique(list(df.columns))
    return df

def infer_binary_target(df: pd.DataFrame, target: str) -> np.ndarray:
    y = df[target]
    if isinstance(y, pd.DataFrame):
        # defensive: if target selection returns DF, take first col
        y = y.iloc[:, 0]

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
        "logloss": float(log_loss(y_true, np.c_[1 - y_prob, y_prob], labels=[0, 1])),
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

def as_datetime_or_year(s):
    # ถ้า df[time_col] ดันได้ DataFrame -> เอาคอลัมน์แรก
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    # ลองแปลงเป็นตัวเลขก่อน (กันกรณี dtype=object แต่จริง ๆ เป็นปี)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        return num

    # ไม่ใช่ปี -> ค่อย parse เป็น datetime
    return pd.to_datetime(s, errors="coerce", utc=True)

def _dedup_list_keep_order(xs: List[str]) -> List[str]:
    # preserves order
    return list(dict.fromkeys(xs))

def _dedup_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    # if any duplicate column labels exist, keep first occurrence
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def _safe_select(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Safe selection that:
    - deduplicates requested cols (order preserved)
    - selects
    - then ensures df has unique col labels (keeps first)
    """
    cols_u = _dedup_list_keep_order([c for c in cols if c in df.columns])
    out = df[cols_u].copy()
    out = _dedup_df_columns(out)
    return out

def _effective_n_splits(y: np.ndarray, requested: int) -> int:
    # Reduce splits if minority class too small; if <2 return 0 to signal "cannot CV"
    y = np.asarray(y).astype(int)
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) < 2:
        return 0
    min_class = int(counts.min())
    return int(min(requested, min_class))


# --------------------------
# Preprocessing / Models
# --------------------------

def build_preprocessor(df: pd.DataFrame, target: str, drop_cols: List[str]) -> ColumnTransformer:
    cols = [c for c in df.columns if c != target and c not in drop_cols]

    def _series_for_col(df_: pd.DataFrame, c: str):
        x = df_[c]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        return x

    cat_cols: List[str] = []
    for c in cols:
        s = _series_for_col(df, c)
        dt = s.dtype
        if dt == "object" or str(dt).startswith("category"):
            cat_cols.append(c)

    num_cols = [c for c in cols if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def base_models(seed: int) -> Dict[str, BaseEstimator]:
    return {
        "LR": LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=800, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "HGB": HistGradientBoostingClassifier(random_state=seed, learning_rate=0.05, max_depth=6, max_iter=800),
    }

# ============================================================
# (4) Temporal split + drift
# ============================================================

def temporal_holdout(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    model: BaseEstimator,
    seed: int,
    train_frac: float = 0.7,
    drop_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    drop_cols = list(drop_cols or [])
    if time_col not in drop_cols:
        drop_cols.append(time_col)

    t = as_datetime_or_year(df[time_col])
    ok = t.notna()
    d = df.loc[ok].copy()
    t = t.loc[ok]

    # order by time
    order = np.argsort(t.to_numpy())
    d = d.iloc[order].reset_index(drop=True)
    t = t.iloc[order].reset_index(drop=True)

    n = len(d)
    cut = int(train_frac * n)
    tr = np.arange(0, cut)
    te = np.arange(cut, n)

    y = infer_binary_target(d, target)
    X = d.drop(columns=[target])
    pre = build_preprocessor(d, target, drop_cols=drop_cols)
    pipe = Pipeline([("pre", pre), ("clf", clone(model))])
    pipe.fit(X.iloc[tr], y[tr])
    p = predict_proba_1(pipe, X.iloc[te])
    return {
        "n_total": int(n),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_frac": float(train_frac),
        "time_min": str(t.min()),
        "time_max": str(t.max()),
        "time_cut": str(t.iloc[cut]) if cut < len(t) else str(t.iloc[-1]),
        "metrics": metric_bundle(y[te], p),
    }

def yearwise_drift(df: pd.DataFrame, target: str, time_col: str) -> pd.DataFrame:
    t = as_datetime_or_year(df[time_col])
    if not pd.api.types.is_numeric_dtype(t):
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.year
    y = infer_binary_target(df, target)
    out = pd.DataFrame({"year": t, "y": y}).dropna()
    grp = out.groupby("year", as_index=False).agg(n=("y","size"), prevalence=("y","mean"))
    return grp.sort_values("year")

# ============================================================
# (5) Proper learning curves for small demo
# ============================================================

def stratified_cv_oof(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    seed: int,
    n_splits: int = 5,
    drop_cols: Optional[List[str]] = None
) -> Dict[str, float]:
    drop_cols = list(drop_cols or [])

    # Ensure unique columns in df (defensive)
    df = _dedup_df_columns(df)

    y = infer_binary_target(df, target)
    eff = _effective_n_splits(y, n_splits)
    if eff < 2:
        # cannot do stratified CV -> return NaNs, but don't crash
        return {"auroc": np.nan, "auprc": np.nan, "logloss": np.nan, "brier": np.nan}

    X = df.drop(columns=[target])
    pre = build_preprocessor(df, target, drop_cols=drop_cols)
    skf = StratifiedKFold(n_splits=eff, shuffle=True, random_state=seed)

    prob = np.full(len(df), np.nan, dtype=float)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = predict_proba_1(pipe, X.iloc[te])

    return metric_bundle(y, prob)

def learning_curve_matched(
    df_demo: pd.DataFrame,
    df_full: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    seed: int,
    n_grid: List[int],
    n_reps: int,
    n_splits: int,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = list(drop_cols or [])
    rows = []
    n_demo_total = len(df_demo)

    # Defensive: ensure unique col labels
    df_demo = _dedup_df_columns(df_demo)
    df_full = _dedup_df_columns(df_full)

    for n in n_grid:
        n_eff = min(int(n), int(n_demo_total))
        for r in range(int(n_reps)):
            rs = seed + 991*r + 11 + n_eff
            demo_s = df_demo.sample(n=n_eff, random_state=rs).reset_index(drop=True)
            full_s = df_full.sample(n=n_eff, random_state=rs).reset_index(drop=True)

            m_demo = stratified_cv_oof(demo_s, target, model, seed=rs, n_splits=n_splits, drop_cols=drop_cols)
            m_full = stratified_cv_oof(full_s, target, model, seed=rs, n_splits=n_splits, drop_cols=drop_cols)

            rows.append({
                "n": int(n_eff), "rep": int(r+1),
                "demo_auroc": m_demo["auroc"], "full_auroc": m_full["auroc"],
                "delta": (m_full["auroc"] - m_demo["auroc"]) if (not np.isnan(m_full["auroc"]) and not np.isnan(m_demo["auroc"])) else np.nan,
            })

    df = pd.DataFrame(rows)
    summ = df.groupby("n", as_index=False).agg(
        delta_mean=("delta","mean"),
        delta_std=("delta","std"),
        demo_mean=("demo_auroc","mean"),
        full_mean=("full_auroc","mean"),
    )
    return df, summ

# ============================================================
# (6) Cross-edition transfer
# ============================================================

def fit_on_source_eval_target(
    source: pd.DataFrame,
    target_df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    seed: int,
    drop_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    drop_cols = list(drop_cols or [])
    source = _dedup_df_columns(source)
    target_df = _dedup_df_columns(target_df)

    y_s = infer_binary_target(source, target)
    X_s = source.drop(columns=[target])
    y_t = infer_binary_target(target_df, target)
    X_t = target_df.drop(columns=[target])

    pre = build_preprocessor(source, target, drop_cols=drop_cols)
    pipe = Pipeline([("pre", pre), ("clf", clone(model))])
    pipe.fit(X_s, y_s)
    p_t = predict_proba_1(pipe, X_t)
    p_s = predict_proba_1(pipe, X_s)
    return {
        "train_metrics_on_source": metric_bundle(y_s, p_s),
        "eval_metrics_on_target": metric_bundle(y_t, p_t),
        "n_source": int(len(source)),
        "n_target": int(len(target_df)),
    }

# ============================================================
# (7) Group-aware bootstrap CI for AUROC gap
# ============================================================

def group_bootstrap_gap_ci(
    df_demo: pd.DataFrame,
    df_full: pd.DataFrame,
    target: str,
    group_col: str,
    model: BaseEstimator,
    seed: int,
    B: int = 500,
    n_splits: int = 5,
    drop_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    drop_cols = list(drop_cols or [])
    df_demo = _dedup_df_columns(df_demo)
    df_full = _dedup_df_columns(df_full)

    if group_col in df_demo.columns and group_col not in drop_cols:
        drop_cols.append(group_col)
    if group_col in df_full.columns and group_col not in drop_cols:
        drop_cols.append(group_col)

    def group_holdout_auroc(df: pd.DataFrame, seed_i: int) -> float:
        if group_col not in df.columns:
            return float("nan")
        y = infer_binary_target(df, target)
        X = df.drop(columns=[target])
        groups = df[group_col].to_numpy()
        pre = build_preprocessor(df, target, drop_cols=drop_cols)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_i)
        tr, te = next(iter(gss.split(X, y, groups=groups)))
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        p = predict_proba_1(pipe, X.iloc[te])
        if len(np.unique(y[te])) < 2:
            return float("nan")
        return float(roc_auc_score(y[te], p))

    gaps = []
    for b in range(int(B)):
        s = seed + 17*b
        a = group_holdout_auroc(df_demo, s)
        c = group_holdout_auroc(df_full, s + 999)
        if not (np.isnan(a) or np.isnan(c)):
            gaps.append(c - a)

    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) == 0:
        return {"B": int(B), "n_effective": 0, "gap_mean": None}

    return {
        "B": int(B),
        "n_effective": int(len(gaps)),
        "gap_mean": float(np.mean(gaps)),
        "gap_ci_95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }

# ============================================================
# (8) Subgroup audits
# ============================================================

def subgroup_auroc(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    seed: int,
    subgroup_col: str,
    min_n: int = 80,
    drop_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    drop_cols = list(drop_cols or [])
    df = _dedup_df_columns(df)

    out = []
    if subgroup_col not in df.columns:
        return pd.DataFrame(out)

    for g, d in df.groupby(subgroup_col):
        if len(d) < int(min_n):
            continue
        y = infer_binary_target(d, target)
        if len(np.unique(y)) < 2:
            continue

        # use safe splits (avoid warning/crash)
        m = stratified_cv_oof(d, target, model, seed=seed, n_splits=5, drop_cols=drop_cols)
        out.append({"subgroup_col": subgroup_col, "group": str(g), "n": int(len(d)), **m})

    return pd.DataFrame(out)

# --------------------------
# Main runner
# --------------------------

def run_all(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir); safe_makedirs(outdir)
    demo = load_csv(args.demo)
    full = load_csv(args.full)

    # extra defensive: ensure unique columns after loading
    demo = _dedup_df_columns(demo)
    full = _dedup_df_columns(full)

    drop_cols = list(args.drop_cols or [])
    id_like = {"subject_id", "hadm_id", "stay_id", "icustay_id"}
    drop_cols = list(set(drop_cols) | id_like)

    # common feature space
    common = sorted(list((set(demo.columns) & set(full.columns)) - {args.target} - set(drop_cols)))
    common = _dedup_list_keep_order(common)  # defensive

    keep_demo = [*common, args.target]
    if args.group_col and args.group_col in demo.columns:
        keep_demo.append(args.group_col)
    if args.time_col and args.time_col in demo.columns:
        keep_demo.append(args.time_col)
    keep_demo = _dedup_list_keep_order(keep_demo)

    keep_full = [*common, args.target]
    if args.group_col and args.group_col in full.columns:
        keep_full.append(args.group_col)
    if args.time_col and args.time_col in full.columns:
        keep_full.append(args.time_col)
    keep_full = _dedup_list_keep_order(keep_full)

    demo2 = _safe_select(demo, keep_demo)
    full2 = _safe_select(full, keep_full)

    model = base_models(args.seed)[args.model]

    summary: Dict[str, Any] = {
        "target": args.target,
        "model": args.model,
        "common_feature_count": int(len(common)),
        "demo_n": int(len(demo2)),
        "full_n": int(len(full2)),
        "warnings": [],
    }

    # 4) temporal
    if args.time_col and args.time_col in demo2.columns and args.time_col in full2.columns:
        summary["demo_temporal_holdout"] = temporal_holdout(
            demo2, args.target, args.time_col, model, args.seed,
            train_frac=args.train_frac,
            drop_cols=[args.group_col] if args.group_col else []
        )
        summary["full_temporal_holdout"] = temporal_holdout(
            full2, args.target, args.time_col, model, args.seed,
            train_frac=args.train_frac,
            drop_cols=[args.group_col] if args.group_col else []
        )
        ydemo = yearwise_drift(demo2, args.target, args.time_col)
        yfull = yearwise_drift(full2, args.target, args.time_col)
        ydemo.to_csv(outdir / "demo_yearwise_prevalence.csv", index=False)
        yfull.to_csv(outdir / "full_yearwise_prevalence.csv", index=False)

        plt.figure(figsize=(8.5, 4.8))
        plt.plot(yfull["year"], yfull["prevalence"], marker="o")
        plt.title("Full: year-wise label prevalence drift")
        plt.xlabel("year")
        plt.ylabel("prevalence")
        plt.tight_layout()
        plt.savefig(outdir / "full_yearwise_prevalence.png", dpi=220)
        plt.close()
    else:
        summary["warnings"].append("time_col missing in one/both datasets; temporal tests skipped.")

    # 5) learning curve
    n_demo = len(demo2)
    if args.n_grid:
        n_grid = [int(x) for x in args.n_grid]
    else:
        n_grid = sorted(set([
            max(30, int(n_demo*0.2)),
            max(50, int(n_demo*0.4)),
            max(80, int(n_demo*0.6)),
            max(120, int(n_demo*0.8)),
            n_demo
        ]))

    # IMPORTANT: use safe selection (prevents duplicate col labels)
    demo_lc = _safe_select(demo2, [*common, args.target])
    full_lc = _safe_select(full2, [*common, args.target])

    df_lc, lc_summ = learning_curve_matched(
        demo_lc, full_lc,
        args.target, model, args.seed,
        n_grid, args.n_reps, args.cv_splits,
        drop_cols=[]
    )
    df_lc.to_csv(outdir / "learning_curve_table_proper.csv", index=False)
    lc_summ.to_csv(outdir / "learning_curve_summary_proper.csv", index=False)

    plt.figure(figsize=(8.5, 4.8))
    plt.errorbar(lc_summ["n"], lc_summ["delta_mean"], yerr=lc_summ["delta_std"], marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.title("Proper learning curve: ΔAUROC(full - demo) vs n")
    plt.xlabel("n (matched)")
    plt.ylabel("Δ AUROC")
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve_delta_auroc_proper.png", dpi=220)
    plt.close()

    # 6) transfer
    tr_demo_to_full = fit_on_source_eval_target(demo_lc, full_lc, args.target, model, args.seed)
    tr_full_to_demo = fit_on_source_eval_target(full_lc, demo_lc, args.target, model, args.seed)
    save_json(outdir / "transfer_demo_to_full.json", tr_demo_to_full)
    save_json(outdir / "transfer_full_to_demo.json", tr_full_to_demo)
    summary["transfer_demo_to_full"] = tr_demo_to_full["eval_metrics_on_target"]
    summary["transfer_full_to_demo"] = tr_full_to_demo["eval_metrics_on_target"]

    # 7) bootstrap CI (if group available)
    if args.group_col and args.group_col in demo2.columns and args.group_col in full2.columns:
        summary["group_bootstrap_gap_ci"] = group_bootstrap_gap_ci(
            demo2, full2, args.target, args.group_col,
            model, args.seed, B=args.bootstrap_B, n_splits=args.cv_splits
        )
    else:
        summary["warnings"].append("group_col missing in one/both datasets; group-bootstrap CI skipped.")

    # 8) subgroup
    subs = []
    for col in args.subgroup_cols:
        demo_sub_cols = [*common, args.target] + ([col] if col in demo2.columns else [])
        full_sub_cols = [*common, args.target] + ([col] if col in full2.columns else [])
        subs.append(subgroup_auroc(_safe_select(demo2, demo_sub_cols), args.target, model, args.seed, col, min_n=args.subgroup_min_n))
        subs.append(subgroup_auroc(_safe_select(full2, full_sub_cols), args.target, model, args.seed, col, min_n=max(args.subgroup_min_n, 200)))

    subdf = pd.concat([d for d in subs if len(d)], ignore_index=True) if any(len(d) for d in subs) else pd.DataFrame([])
    if len(subdf):
        subdf.to_csv(outdir / "subgroup_auroc_audit.csv", index=False)
    else:
        summary["warnings"].append("No subgroup columns produced enough samples; subgroup audit empty.")

    save_json(outdir / "final_boosters_summary.json", summary)
    print(f"[done] final boosters written to {outdir.resolve()}")

# --------------------------
# CLI
# --------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICML Stability Final Boosters")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run all final boosters")
    r.add_argument("--demo", required=True)
    r.add_argument("--full", required=True)
    r.add_argument("--target", required=True)
    r.add_argument("--model", default="HGB", choices=["LR","RF","HGB"])
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--group-col", default="subject_id")
    r.add_argument("--time-col", default="anchor_year")
    r.add_argument("--train-frac", type=float, default=0.7)
    r.add_argument("--cv-splits", type=int, default=5)
    r.add_argument("--n-reps", type=int, default=30)
    r.add_argument("--n-grid", nargs="*", default=None, help="Optional explicit grid; e.g., 50 100 150 200 252")
    r.add_argument("--bootstrap-B", type=int, default=500)
    r.add_argument("--subgroup-cols", nargs="*", default=["gender","race","insurance","admission_type"])
    r.add_argument("--subgroup-min-n", type=int, default=80)
    r.add_argument("--drop-cols", nargs="*", default=[])
    r.add_argument("--outdir", required=True)

    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    outdir = Path(args.outdir); safe_makedirs(outdir)
    save_json(outdir / "run_args.json", vars(args))

    if args.cmd == "run":
        run_all(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()
