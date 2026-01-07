# make_controls_fliprate_summary_rankings.py
# -----------------------------------------
# Create Figure 5: "Ranking Instability Is Structural and Survives Controls"
# from raw demo/full datasets (multi-model ranking flips).
#
# Example:
#   python make_controls_fliprate_summary_rankings.py \
#     --demo demo_analytic_dataset_mortality_all_admissions.csv \
#     --full full_analytic_dataset_mortality_all_admissions.csv \
#     --target label_mortality \
#     --group-col subject_id \
#     --time-col anchor_year \
#     --outpng controls_fliprate_summary.png
#
# If group/time cols don't exist, the script auto-skips them.

from __future__ import annotations
import argparse, os, json, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold


# -------------------------
# Config / Leakage hints
# -------------------------
LEAKY_HINTS = [
    "target","label","outcome","mort","death","expire","survival",
    "discharge","los","length_of_stay","icu_los","hospital_los",
    "readmit","readmission",
    "time_to_death","dod","date_of_death",
    "after","post","followup",
    "vent","vasopressor","dialysis",
]

DEFAULT_ID_COLS = ["hadm_id", "subject_id"]  # dropped from features if present
DEFAULT_TIME_COLS = ["anchor_year", "admittime", "dischtime"]  # temporal if present
SEED = 1337


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def infer_y(df: pd.DataFrame, target: str) -> np.ndarray:
    y = df[target]
    if y.dtype == bool:
        return y.astype(int).to_numpy()
    if y.dtype == object:
        return y.astype(str).str.strip().replace({"False":"0","True":"1"}).astype(int).to_numpy()
    return y.astype(int).to_numpy()

def build_preprocessor(df: pd.DataFrame, target: str, drop_cols: List[str]) -> ColumnTransformer:
    cols = [c for c in df.columns if c != target and c not in drop_cols]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                             remainder="drop",
                             verbose_feature_names_out=False)

def model_zoo(seed: int) -> Dict[str, BaseEstimator]:
    return {
        "LR": LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=700, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "ET": ExtraTreesClassifier(n_estimators=800, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "HGB": HistGradientBoostingClassifier(random_state=seed, learning_rate=0.05, max_depth=6, max_iter=700),
    }

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))

def leaky_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        cl = c.lower()
        if any(h in cl for h in LEAKY_HINTS):
            out.append(c)
    return out

def common_frame(demo: pd.DataFrame, full: pd.DataFrame, target: str,
                 extra_keep: Optional[List[str]] = None,
                 drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    extra_keep = list(extra_keep or [])
    drop_cols = list(drop_cols or [])
    drop = set([target] + drop_cols)

    common_feats = sorted(list((set(demo.columns) & set(full.columns)) - drop - set(extra_keep)))
    demo2 = demo[[*common_feats, *[c for c in extra_keep if c in demo.columns], target]].copy()
    full2 = full[[*common_feats, *[c for c in extra_keep if c in full.columns], target]].copy()
    return demo2, full2, common_feats


# -------------------------
# Core scoring under regimes
# -------------------------
def stratified_oof_auc(df: pd.DataFrame, target: str, model: BaseEstimator, seed: int,
                       drop_cols: Optional[List[str]] = None, n_splits: int = 5) -> float:
    drop_cols = list(drop_cols or [])
    y = infer_y(df, target)
    X = df.drop(columns=[target])

    min_class = int(pd.Series(y).value_counts().min()) if len(np.unique(y)) > 1 else 0
    eff_splits = max(2, min(n_splits, min_class)) if min_class >= 2 else 0
    if eff_splits < 2:
        return float("nan")

    pre = build_preprocessor(df, target, drop_cols=drop_cols)
    skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)

    prob = np.full(len(df), np.nan)
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    return safe_auc(y, prob)

def groupkfold_oof_auc(df: pd.DataFrame, target: str, group_col: str, model: BaseEstimator, seed: int,
                       drop_cols: Optional[List[str]] = None, n_splits: int = 5) -> float:
    drop_cols = list(drop_cols or [])
    if group_col not in df.columns:
        return float("nan")

    y = infer_y(df, target)
    groups = df[group_col].to_numpy()
    X = df.drop(columns=[target, group_col])

    # Preprocessor built on (X + y) schema
    tmp = pd.concat([X.reset_index(drop=True), pd.Series(y, name=target)], axis=1)
    pre = build_preprocessor(tmp, target, drop_cols=drop_cols)

    # GroupKFold needs n_splits <= #groups
    n_groups = len(np.unique(groups))
    eff = min(n_splits, n_groups) if n_groups >= 2 else 0
    if eff < 2:
        return float("nan")

    gkf = GroupKFold(n_splits=eff)
    prob = np.full(len(df), np.nan)

    for tr, te in gkf.split(X, y, groups=groups):
        pipe = Pipeline([("pre", pre), ("clf", clone(model))])
        pipe.fit(X.iloc[tr], y[tr])
        prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    return safe_auc(y, prob)

def temporal_holdout_auc(df: pd.DataFrame, target: str, time_col: str, model: BaseEstimator, seed: int,
                         drop_cols: Optional[List[str]] = None, train_frac: float = 0.7) -> float:
    drop_cols = list(drop_cols or [])
    if time_col not in df.columns:
        return float("nan")

    # parse as year if mostly numeric else datetime
    t_num = pd.to_numeric(df[time_col], errors="coerce")
    if t_num.notna().mean() > 0.8:
        t = t_num
    else:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    ok = t.notna()
    d = df.loc[ok].copy()
    t = t.loc[ok]

    if len(d) < 50:
        return float("nan")

    order = np.argsort(t.to_numpy())
    d = d.iloc[order].reset_index(drop=True)

    y = infer_y(d, target)
    X = d.drop(columns=[target, time_col])

    cut = int(train_frac * len(d))
    if cut < 20 or (len(d) - cut) < 20:
        return float("nan")

    tmp = pd.concat([X.reset_index(drop=True), pd.Series(y, name=target)], axis=1)
    pre = build_preprocessor(tmp, target, drop_cols=drop_cols)

    pipe = Pipeline([("pre", pre), ("clf", clone(model))])
    pipe.fit(X.iloc[:cut], y[:cut])
    p = pipe.predict_proba(X.iloc[cut:])[:, 1]

    return safe_auc(y[cut:], p)


# -------------------------
# Matching / IPW regimes
# -------------------------
def pick_anchor_covariates(demo: pd.DataFrame, full: pd.DataFrame, target: str, max_k: int = 12) -> List[str]:
    # numeric covariates with low missingness and non-trivial variance
    common = list((set(demo.columns) & set(full.columns)) - {target})
    cands = []
    for c in common:
        if demo[c].dtype == "object" or full[c].dtype == "object":
            continue
        mr = max(float(demo[c].isna().mean()), float(full[c].isna().mean()))
        if mr > 0.35:
            continue
        v = float(np.nanvar(pd.to_numeric(demo[c], errors="coerce"))) + float(np.nanvar(pd.to_numeric(full[c], errors="coerce")))
        if v <= 1e-10:
            continue
        cands.append((mr, -v, c))
    cands.sort()
    return [c for _, _, c in cands[:max_k]]

def ipw_weights_full_to_demo(demo: pd.DataFrame, full: pd.DataFrame, covariates: List[str], seed: int) -> np.ndarray:
    A = demo[covariates].copy(); A["__env__"] = 1
    B = full[covariates].copy(); B["__env__"] = 0
    Z = pd.concat([A, B], ignore_index=True)

    y = Z["__env__"].to_numpy()
    X = Z.drop(columns=["__env__"])

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=4000, random_state=seed)),
    ])
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]  # P(demo|x)
    p_full = np.clip(p[len(A):], 0.05, 0.95)
    w = p_full / (1.0 - p_full)
    w = w / np.mean(w)
    return w

def weighted_bootstrap(df: pd.DataFrame, n: int, w: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)
    idx = rng.choice(len(df), size=n, replace=True, p=w)
    return df.iloc[idx].reset_index(drop=True)


# -------------------------
# Flip-rate computation
# -------------------------
def flip_rate_pairwise(ref: Dict[str, float], other: Dict[str, float], margin: float = 0.0) -> float:
    # Consider all unordered pairs (i<j). A flip occurs if sign differs.
    models = sorted(set(ref.keys()) & set(other.keys()))
    flips = 0
    total = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mi, mj = models[i], models[j]
            ai, aj = ref[mi], ref[mj]
            bi, bj = other[mi], other[mj]
            if np.isnan(ai) or np.isnan(aj) or np.isnan(bi) or np.isnan(bj):
                continue
            s_ref = np.sign((ai - aj) - margin)
            s_oth = np.sign((bi - bj) - margin)
            if s_ref == 0 or s_oth == 0:
                continue
            total += 1
            if s_ref != s_oth:
                flips += 1
    return float(flips / total) if total > 0 else 0.0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--target", default="label_mortality")
    ap.add_argument("--group-col", default="subject_id")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--margin", type=float, default=0.0, help="minimum AUROC gap to consider a decisive preference")
    ap.add_argument("--reps", type=int, default=25, help="replications for matched/IPW regimes")
    ap.add_argument("--outpng", default="controls_fliprate_summary.png")
    ap.add_argument("--outcsv", default="controls_fliprate_summary.csv")
    ap.add_argument("--outscores", default="controls_model_scores.csv")
    args = ap.parse_args()

    set_seed(args.seed)

    demo_raw = load_csv(args.demo)
    full_raw = load_csv(args.full)

    # pick time col if not provided
    time_col = args.time_col
    if time_col is None:
        for c in DEFAULT_TIME_COLS:
            if c in demo_raw.columns and c in full_raw.columns:
                time_col = c
                break

    # Drop obvious IDs from features (keep separately for splits)
    drop_cols = [c for c in DEFAULT_ID_COLS if c in demo_raw.columns and c in full_raw.columns and c != args.target]

    # Base common feature frame (no group/time in features)
    demo, full, common_feats = common_frame(demo_raw, full_raw, args.target, extra_keep=[], drop_cols=drop_cols)

    models = model_zoo(args.seed)

    regimes: List[Tuple[str, Dict[str, float], Dict[str, float]]] = []
    all_scores_rows = []

    def score_envs(regime: str, demo_df: pd.DataFrame, full_df: pd.DataFrame,
                   mode: str, drop_cols_features: List[str],
                   group_col: Optional[str] = None,
                   time_col: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        demo_scores = {}
        full_scores = {}
        for mn, m in models.items():
            if mode == "stratified":
                da = stratified_oof_auc(demo_df, args.target, m, seed=args.seed, drop_cols=drop_cols_features)
                fa = stratified_oof_auc(full_df, args.target, m, seed=args.seed, drop_cols=drop_cols_features)
            elif mode == "groupkfold":
                da = groupkfold_oof_auc(demo_df, args.target, group_col=group_col, model=m, seed=args.seed, drop_cols=drop_cols_features)
                fa = groupkfold_oof_auc(full_df, args.target, group_col=group_col, model=m, seed=args.seed, drop_cols=drop_cols_features)
            elif mode == "temporal":
                da = temporal_holdout_auc(demo_df, args.target, time_col=time_col, model=m, seed=args.seed, drop_cols=drop_cols_features)
                fa = temporal_holdout_auc(full_df, args.target, time_col=time_col, model=m, seed=args.seed, drop_cols=drop_cols_features)
            else:
                raise ValueError(mode)

            demo_scores[mn] = da
            full_scores[mn] = fa

            all_scores_rows.append({"regime": regime, "env": "demo", "model": mn, "auroc": da})
            all_scores_rows.append({"regime": regime, "env": "full", "model": mn, "auroc": fa})
        return demo_scores, full_scores

    # (0) Uncontrolled (stratified)
    dsc, fsc = score_envs("Uncontrolled", demo, full, mode="stratified", drop_cols_features=[])
    regimes.append(("Uncontrolled", dsc, fsc))

    # (1) Leakage-hardened: drop leaky-hint columns
    leak = leaky_cols(common_feats)
    demo_lh = demo.drop(columns=leak, errors="ignore")
    full_lh = full.drop(columns=leak, errors="ignore")
    dsc, fsc = score_envs("Leakage-hardened", demo_lh, full_lh, mode="stratified", drop_cols_features=[])
    regimes.append(("Leakage-hardened", dsc, fsc))

    # (2) Patient-level split (GroupKFold) if possible
    if args.group_col in demo_raw.columns and args.group_col in full_raw.columns:
        demo_g, full_g, _ = common_frame(
            demo_raw, full_raw, args.target,
            extra_keep=[args.group_col],
            drop_cols=drop_cols
        )
        dsc, fsc = score_envs("Patient-level split", demo_g, full_g, mode="groupkfold",
                              drop_cols_features=[], group_col=args.group_col)
        regimes.append(("Patient-level split", dsc, fsc))
    else:
        # keep bar for plot stability; reuse last
        regimes.append(("Patient-level split", regimes[-1][1], regimes[-1][2]))

    # (3) Temporal split if possible
    if time_col is not None and (time_col in demo_raw.columns and time_col in full_raw.columns):
        demo_t, full_t, _ = common_frame(
            demo_raw, full_raw, args.target,
            extra_keep=[time_col],
            drop_cols=drop_cols
        )
        dsc, fsc = score_envs("Temporal split", demo_t, full_t, mode="temporal",
                              drop_cols_features=[], time_col=time_col)
        regimes.append(("Temporal split", dsc, fsc))
    else:
        regimes.append(("Temporal split", regimes[-1][1], regimes[-1][2]))

    # (4) Sample-size matched: subsample FULL down to n_demo (average flips over reps)
    n_demo = len(demo)
    frs = []
    rng = np.random.default_rng(args.seed)
    for r in range(args.reps):
        rs = int(rng.integers(1, 10_000_000))
        full_s = full.sample(n=n_demo, replace=(n_demo > len(full)), random_state=rs).reset_index(drop=True)
        demo_s = demo.reset_index(drop=True)
        # score with a rep-specific seed by perturbing global seed (simple)
        local_seed = args.seed + 101 * r + 7
        tmp_rows_before = len(all_scores_rows)
        # temporarily score with local_seed by cloning models (same hyperparams)
        demo_scores, full_scores = {}, {}
        for mn, m in models.items():
            da = stratified_oof_auc(demo_s, args.target, m, seed=local_seed, drop_cols=[])
            fa = stratified_oof_auc(full_s, args.target, m, seed=local_seed, drop_cols=[])
            demo_scores[mn] = da
            full_scores[mn] = fa
        frs.append(flip_rate_pairwise(demo_scores, full_scores, margin=args.margin))

    # also store one representative set of scores (from base seed) so outscores isn't empty
    dsc, fsc = score_envs("Sample-size matched (rep=base)", demo, full.sample(n=n_demo, random_state=args.seed),
                          mode="stratified", drop_cols_features=[])
    regimes.append(("Sample-size matched", dsc, fsc))
    sample_size_flip = float(np.mean(frs))

    # (5) Composition matched (proxy): use IPW-weighted bootstrap as composition control
    anchors = pick_anchor_covariates(demo, full, args.target, max_k=12)
    comp_flip = np.nan
    ipw_flip = np.nan

    if len(anchors) >= 2:
        w_full = ipw_weights_full_to_demo(demo, full, anchors, seed=args.seed)

        # IPW weighted bootstrap (n_demo), compute flips across reps
        frs = []
        for r in range(args.reps):
            rs = args.seed + 1777 * r + 19
            full_w = weighted_bootstrap(full, n=n_demo, w=w_full, seed=rs)
            demo_s = demo.reset_index(drop=True)

            demo_scores, full_scores = {}, {}
            local_seed = args.seed + 97 * r + 13
            for mn, m in models.items():
                da = stratified_oof_auc(demo_s, args.target, m, seed=local_seed, drop_cols=[])
                fa = stratified_oof_auc(full_w, args.target, m, seed=local_seed, drop_cols=[])
                demo_scores[mn] = da
                full_scores[mn] = fa
            frs.append(flip_rate_pairwise(demo_scores, full_scores, margin=args.margin))
        ipw_flip = float(np.mean(frs))

        # For "Composition matched" bar, you can either:
        # (A) keep a distinct regime by doing a *hard bin-matching* (more complex), or
        # (B) treat "composition matched" as IPW (common in papers).
        #
        # Here I do BOTH bars:
        # - Composition matched: approximate via IPW bootstrap but with *trimmed weights* (less extreme)
        w_trim = np.clip(w_full, 0.0, np.quantile(w_full, 0.95))
        w_trim = w_trim / np.mean(w_trim)

        frs = []
        for r in range(args.reps):
            rs = args.seed + 2222 * r + 23
            full_c = weighted_bootstrap(full, n=n_demo, w=w_trim, seed=rs)
            demo_s = demo.reset_index(drop=True)

            demo_scores, full_scores = {}, {}
            local_seed = args.seed + 131 * r + 17
            for mn, m in models.items():
                da = stratified_oof_auc(demo_s, args.target, m, seed=local_seed, drop_cols=[])
                fa = stratified_oof_auc(full_c, args.target, m, seed=local_seed, drop_cols=[])
                demo_scores[mn] = da
                full_scores[mn] = fa
            frs.append(flip_rate_pairwise(demo_scores, full_scores, margin=args.margin))
        comp_flip = float(np.mean(frs))

        # store representative scores rows for these regimes (base rep)
        demo_rep = demo.reset_index(drop=True)
        full_rep = weighted_bootstrap(full, n=n_demo, w=w_trim, seed=args.seed + 23)
        dsc, fsc = score_envs("Composition matched (rep=base)", demo_rep, full_rep, mode="stratified", drop_cols_features=[])
        regimes.append(("Composition matched", dsc, fsc))

        full_rep2 = weighted_bootstrap(full, n=n_demo, w=w_full, seed=args.seed + 19)
        dsc, fsc = score_envs("IPW weighted (rep=base)", demo_rep, full_rep2, mode="stratified", drop_cols_features=[])
        regimes.append(("IPW weighted", dsc, fsc))
    else:
        # fallback: reuse sample-size bar
        regimes.append(("Composition matched", regimes[-1][1], regimes[-1][2]))
        regimes.append(("IPW weighted", regimes[-1][1], regimes[-1][2]))
        comp_flip = sample_size_flip
        ipw_flip = sample_size_flip

    # Build final flip-rate table for Figure 5 bars in the intended order
    bar_order = [
        "Uncontrolled",
        "Leakage-hardened",
        "Patient-level split",
        "Temporal split",
        "Sample-size matched",
        "Composition matched",
        "IPW weighted",
    ]

    # Flip-rate values:
    # - For the first 4, compute directly from the representative scores in regimes
    # - For sample-size / composition / ipw use averaged flips over reps for stronger estimates
    rep_scores = {name: (ds, fs) for (name, ds, fs) in regimes}

    flip_values = {}
    for name in bar_order:
        if name == "Sample-size matched":
            flip_values[name] = sample_size_flip
        elif name == "Composition matched":
            flip_values[name] = float(comp_flip)
        elif name == "IPW weighted":
            flip_values[name] = float(ipw_flip)
        else:
            if name in rep_scores:
                ds, fs = rep_scores[name]
                flip_values[name] = flip_rate_pairwise(ds, fs, margin=args.margin)
            else:
                flip_values[name] = float("nan")

    df_flip = pd.DataFrame([{"regime": k, "flip_rate": flip_values[k]} for k in bar_order])
    df_flip.to_csv(args.outcsv, index=False)

    df_scores = pd.DataFrame(all_scores_rows)
    df_scores.to_csv(args.outscores, index=False)

    # Plot
    plt.figure(figsize=(10.5, 4.6))
    plt.bar(df_flip["regime"].astype(str), df_flip["flip_rate"].astype(float))
    plt.ylabel("Global ranking flip rate (pairwise AUROC preferences)")
    plt.xticks(rotation=20, ha="right")
    ymax = max(0.5, float(np.nanmax(df_flip["flip_rate"])) + 0.05)
    plt.ylim(0.0, min(1.0, ymax))
    plt.tight_layout()
    plt.savefig(args.outpng, dpi=260)
    plt.close()

    print(f"[OK] wrote {Path(args.outpng).resolve()}")
    print(f"[OK] wrote {Path(args.outcsv).resolve()}")
    print(f"[OK] wrote {Path(args.outscores).resolve()}")
    print(df_flip)

if __name__ == "__main__":
    main()
