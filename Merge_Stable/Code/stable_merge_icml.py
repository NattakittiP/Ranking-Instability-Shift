# stable_merge_icml.py
# ============================================================
# STABLE-MERGE: Stability-Targeted Algorithm for Learning &
# Model Selection under Latent Evaluation Shifts
#
# Supports:
#  - MIMIC-IV demo/full mortality classification
#  - Synthetic TG vs NHANES cross-cohort TG response classification
#
# Key Outputs:
#  - Per-env OOF metrics: AUROC, AUPRC, LogLoss, Brier, ECE, Cal slope/intercept
#  - Explanation stability proxy (model-specific)
#  - Ranking stability across envs (Kendall tau)
#  - STABLE-MERGE score + selected model + optional merged ensemble
#
# Notes:
#  - Strictly leakage-controlled: preprocessing inside CV
#  - Calibration measured in-fold (no peeking)
# ============================================================

from __future__ import annotations

import os
import json
import math
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# tqdm (ละเอียด)
from tqdm.auto import tqdm

# optional: XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# optional: SciPy optimizer for stable ensemble weights
try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Kendall tau for ranking stability
try:
    from scipy.stats import kendalltau
    _HAS_KTAU = True
except Exception:
    _HAS_KTAU = False


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------
# Calibration metrics
# ----------------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15,
                               pbar: Optional[tqdm] = None) -> float:
    """
    Standard ECE with equal-width bins in [0,1].
    """
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    it = range(n_bins)
    it = tqdm(it, desc="  ECE bins", leave=False) if pbar is None else it

    for i in it:
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")  # หรือ (1.0, 0.0) ก็ได้ตามที่อยากนิยาม

    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs", max_iter=500)
    lr.fit(logit, y_true)
    return float(lr.coef_.ravel()[0]), float(lr.intercept_.ravel()[0])



# ----------------------------
# Explanation stability proxies
# ----------------------------
def explanation_proxy(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    seed: int = 42,
    n_repeats: int = 5,
    pbar_outer: Optional[tqdm] = None
) -> np.ndarray:
    """
    Returns a vector importance proxy for stability comparisons.

    - For LogisticRegression: absolute standardized coefficients after preprocessing
    - For Tree/Ensembles/MLP: permutation importance on prediction prob
    """
    pre = fitted_pipeline.named_steps["pre"]
    clf = fitted_pipeline.named_steps["clf"]

    # Transform X to numeric matrix
    if pbar_outer is not None:
        pbar_outer.set_postfix(step="pre.transform", refresh=True)
    X_t = pre.transform(X)

    # Importance for logreg: |coef|
    if isinstance(clf, LogisticRegression):
        if pbar_outer is not None:
            pbar_outer.set_postfix(step="coef_importance", refresh=True)
        coef = np.abs(clf.coef_.ravel())
        if coef.sum() > 0:
            coef = coef / (coef.sum() + 1e-12)
        return coef.astype(float)

    # Permutation importance otherwise
    if pbar_outer is not None:
        pbar_outer.set_postfix(step="permutation_importance", refresh=True)

    rng = np.random.RandomState(seed)
    r = permutation_importance(
        clf, X_t, y,
        n_repeats=n_repeats,
        random_state=rng,
        scoring="roc_auc"
    )
    imp = np.maximum(0.0, r.importances_mean)
    if imp.sum() > 0:
        imp = imp / (imp.sum() + 1e-12)
    return imp.astype(float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float); b = b.astype(float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------------------
# Data configs
# ----------------------------
@dataclass
class EnvDataset:
    name: str
    df: pd.DataFrame
    target_col: str
    drop_cols: List[str]


@dataclass
class Metrics:
    auroc: float
    auprc: float
    logloss: float
    brier: float
    ece: float
    cal_slope: float
    cal_intercept: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ModelResult:
    model_name: str
    env_metrics: Dict[str, Metrics]
    env_oof_probs: Dict[str, np.ndarray]
    env_oof_y: Dict[str, np.ndarray]
    env_expl: Dict[str, np.ndarray]
    ranking_score_by_env: Dict[str, float]
    stability: Dict[str, float]
    stable_merge_score: float


# ----------------------------
# Dataset loaders for your files
# ----------------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def build_mimic_envs(
    demo_path: str,
    full_path: str,
    target_col: str = "label_mortality"
) -> List[EnvDataset]:
    demo = load_csv(demo_path)
    full = load_csv(full_path)

    drop_demo = [c for c in ["hadm_id", "subject_id"] if c in demo.columns]
    drop_full = [c for c in ["hadm_id", "subject_id"] if c in full.columns]

    return [
        EnvDataset(name="MIMIC_DEMO", df=demo, target_col=target_col, drop_cols=drop_demo),
        EnvDataset(name="MIMIC_FULL", df=full, target_col=target_col, drop_cols=drop_full),
    ]


def build_tg_envs(
    nhanes_path: str,
    synthetic_path: str,
    task: str = "tg4h_high",
    quantile: float = 0.75
) -> List[EnvDataset]:

    nh = load_csv(nhanes_path)
    sy = load_csv(synthetic_path)

    def add_label(df: pd.DataFrame, env_name: str) -> pd.DataFrame:
        df = df.copy()

        if task == "tg4h_high":
            # เลือกคอลัมน์ TG ที่ "ใช้ได้จริง" (ไม่ใช่มีชื่อแต่ all-missing)
            candidates = []
            if "TG4H" in df.columns:
                candidates.append("TG4H")
            if "TG4h" in df.columns:
                candidates.append("TG4h")
            if "TG" in df.columns:
                candidates.append("TG")

            if not candidates:
                raise ValueError(f"{env_name}: TG4H/TG4h/TG column not found.")

            chosen = None
            s = None
            for col in candidates:
                s_try = pd.to_numeric(df[col], errors="coerce")
                if s_try.notna().any():   # มีค่าจริงอย่างน้อย 1 ตัว
                    chosen = col
                    s = s_try
                    break

            if chosen is None:
                raise ValueError(f"{env_name}: TG columns exist but are all-missing -> cannot create label.")

            # กันค่าติดลบ (พบได้ใน synthetic)
            s = s.clip(lower=0)

            thr = float(s.dropna().quantile(quantile))
            df["label_tg"] = (s >= thr).astype(int)
            df = df.loc[s.notna()].copy()

            # (optional) ถ้าอยากรู้ว่า env นี้ใช้คอลัมน์ไหน:
            # print(f"[{env_name}] tg4h_high uses column: {chosen} (thr={thr:.3f})")

        elif task == "tcr_low":
            if "TCR" not in df.columns:
                raise ValueError(f"{env_name}: TCR column not found.")

            s = pd.to_numeric(df["TCR"], errors="coerce")
            if not s.notna().any():
                raise ValueError(f"{env_name}: TCR is all-missing -> cannot create label.")

            thr = float(s.dropna().quantile(quantile))
            df["label_tg"] = (s <= thr).astype(int)
            df = df.loc[s.notna()].copy()

        else:
            raise ValueError(f"Unknown task: {task}")

        # กันพลาด: ถ้ายังเหลือคลาสเดียว ให้ฟ้องชัด ๆ
        if df["label_tg"].nunique() < 2:
            vc = df["label_tg"].value_counts(dropna=False).to_dict()
            raise ValueError(f"{env_name}: label has only one class after labeling: {vc}")

        return df

    nh = add_label(nh, "NHANES")
    sy = add_label(sy, "SYNTHETIC")

    drop_nh = [c for c in ["SEQN"] if c in nh.columns]
    drop_sy = [c for c in ["ID"] if c in sy.columns]

    return [
        EnvDataset(name="NHANES", df=nh, target_col="label_tg", drop_cols=drop_nh),
        EnvDataset(name="SYNTHETIC", df=sy, target_col="label_tg", drop_cols=drop_sy),
    ]



# ----------------------------
# Preprocessing builder
# ----------------------------
def make_preprocessor(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    X = X.dropna(axis=1, how="all")
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre, num_cols, cat_cols


# ----------------------------
# Model zoo
# ----------------------------
def model_zoo(seed: int = 42) -> Dict[str, BaseEstimator]:
    zoo: Dict[str, BaseEstimator] = {}

    zoo["LogReg_L2"] = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=1.0, class_weight="balanced"
    )

    zoo["RF_500"] = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    zoo["ExtraTrees_800"] = ExtraTreesClassifier(
        n_estimators=800,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    zoo["MLP_256x128"] = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        alpha=1e-4,
        max_iter=600,
        early_stopping=True,
        random_state=seed
    )

    if _HAS_XGB:
        zoo["XGB_600"] = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=2.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss"
        )
    return zoo


# ----------------------------
# OOF evaluation per env
# ----------------------------
def oof_eval_env(
    env: EnvDataset,
    base_model: BaseEstimator,
    n_splits: int,
    seed: int,
    pbar_env: Optional[tqdm] = None
) -> Tuple[Metrics, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      metrics, oof_prob, y_true, explanation_proxy_vector
    """
    df = env.df.copy()
    y = df[env.target_col].astype(int).to_numpy()

    if pbar_env is not None:
        pbar_env.set_postfix(step="make_preprocessor", refresh=True)
    pre, _, _ = make_preprocessor(df, env.target_col, env.drop_cols)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", clone(base_model))
    ])

    vals, counts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        raise ValueError(f"[{env.name}] Target has only one class: {vals[0]} (count={counts[0]}).")

    min_class = counts.min()
    if n_splits > min_class:
        raise ValueError(
            f"[{env.name}] n_splits={n_splits} is too large for the minority class (min_class_count={min_class}). "
            f"Reduce n_splits to <= {min_class}."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(df), dtype=float)

    splits = list(skf.split(df, y))

    # Fold progress (ละเอียด)
    pbar_fold = tqdm(
        enumerate(splits, start=1),
        total=len(splits),
        desc=f"    CV folds [{env.name}]",
        leave=False
    )
    for fold_i, (tr, te) in pbar_fold:
        if pbar_env is not None:
            pbar_env.set_postfix(step=f"fold {fold_i}/{n_splits}", refresh=True)

        Xtr = df.iloc[tr].drop(columns=[env.target_col] + env.drop_cols, errors="ignore")
        ytr = y[tr]
        Xte = df.iloc[te].drop(columns=[env.target_col] + env.drop_cols, errors="ignore")

        fold_pipe = clone(pipe)

        pbar_fold.set_postfix(stage="fit", refresh=True)
        fold_pipe.fit(Xtr, ytr)

        pbar_fold.set_postfix(stage="predict_proba", refresh=True)
        p = fold_pipe.predict_proba(Xte)[:, 1]
        oof_prob[te] = p

    # metrics on OOF
    if pbar_env is not None:
        pbar_env.set_postfix(step="metrics", refresh=True)

    p = np.clip(oof_prob, 1e-12, 1 - 1e-12)

    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    auprc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    logl = float(log_loss(y, p))
    brier = float(brier_score_loss(y, p))

    # ECE แบบละเอียด (bin progress)
    ece = float(expected_calibration_error(y, p, n_bins=15))

    slope, intercept = calibration_slope_intercept(y, p)

    m = Metrics(
        auroc=auroc,
        auprc=auprc,
        logloss=logl,
        brier=brier,
        ece=ece,
        cal_slope=float(slope),
        cal_intercept=float(intercept),
    )

    # Fit on full env for explanation proxy
    if pbar_env is not None:
        pbar_env.set_postfix(step="fit_full_for_expl", refresh=True)

    Xfull = df.drop(columns=[env.target_col] + env.drop_cols, errors="ignore")
    pipe.fit(Xfull, y)

    if pbar_env is not None:
        pbar_env.set_postfix(step="expl_proxy", refresh=True)

    expl = explanation_proxy(pipe, Xfull, y, model_name=type(base_model).__name__, seed=seed, pbar_outer=pbar_env)

    return m, oof_prob, y, expl


# ----------------------------
# Stability + STABLE-MERGE scoring
# ----------------------------
def env_score_for_ranking(metrics: Metrics) -> float:
    """
    A single scalar per env used to rank models (selection integrity).
    Higher is better.

    We reward AUROC and penalize LogLoss + ECE + Brier.
    """
    auroc = 0.0 if not np.isfinite(metrics.auroc) else metrics.auroc
    logl = metrics.logloss
    ece = metrics.ece
    brier = metrics.brier
    return float(0.55 * auroc + 0.25 * math.exp(-logl) - 0.10 * ece - 0.10 * brier)


def kendall_tau_rank_stability(ranks_by_env: Dict[str, List[str]], pbar: Optional[tqdm] = None) -> float:
    """
    Given env->ordered model list (best->worst), compute average Kendall tau across env pairs.
    """
    envs = list(ranks_by_env.keys())
    if len(envs) < 2:
        return 1.0

    model_names = ranks_by_env[envs[0]]
    pos = {e: {m: i for i, m in enumerate(ranks_by_env[e])} for e in envs}

    pairs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            pairs.append((envs[i], envs[j]))

    it = tqdm(pairs, desc="Ranking stability (env pairs)", leave=False) if pbar is None else pairs

    taus = []
    for (e1, e2) in it:
        x = [pos[e1][m] for m in model_names]
        y = [pos[e2][m] for m in model_names]
        if _HAS_KTAU:
            tau = float(kendalltau(x, y).correlation)
        else:
            tau = float(np.corrcoef(np.argsort(x), np.argsort(y))[0, 1])
        if np.isnan(tau):
            tau = 0.0
        taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0


def aggregate_stabilities(env_metrics: Dict[str, Metrics], env_expl: Dict[str, np.ndarray],
                          pbar: Optional[tqdm] = None) -> Dict[str, float]:
    """
    Compute:
      - perf_stability: exp(-3*std(AUROC))
      - cal_stability: exp(-5*std(ECE))
      - expl_stability: avg cosine similarity across env pairs
    """
    envs = list(env_metrics.keys())

    if pbar is not None:
        pbar.set_postfix(step="collect_metrics", refresh=True)

    aurocs = np.array([env_metrics[e].auroc for e in envs], dtype=float)
    eces = np.array([env_metrics[e].ece for e in envs], dtype=float)

    aurocs = np.nan_to_num(aurocs, nan=np.nanmean(aurocs) if np.isfinite(np.nanmean(aurocs)) else 0.0)
    eces = np.nan_to_num(eces, nan=np.nanmean(eces) if np.isfinite(np.nanmean(eces)) else 1.0)

    perf_std = float(np.std(aurocs))
    cal_std = float(np.std(eces))

    # explanation cosine similarities
    pairs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            pairs.append((envs[i], envs[j]))

    if pbar is not None:
        pbar.set_postfix(step="expl_cosine", refresh=True)

    cos = []
    it = tqdm(pairs, desc="  Expl stability (env pairs)", leave=False) if pbar is None else pairs
    for (e1, e2) in it:
        a = env_expl[e1]
        b = env_expl[e2]
        if a.shape[0] != b.shape[0]:
            m = max(a.shape[0], b.shape[0])
            aa = np.zeros(m); bb = np.zeros(m)
            aa[:a.shape[0]] = a
            bb[:b.shape[0]] = b
            a, b = aa, bb
        cos.append(cosine_similarity(a, b))
    expl = float(np.mean(cos)) if cos else 0.0

    perf_stab = float(math.exp(-3.0 * perf_std))
    cal_stab = float(math.exp(-5.0 * cal_std))

    return {
        "perf_std": perf_std,
        "cal_std": cal_std,
        "perf_stability": perf_stab,
        "cal_stability": cal_stab,
        "expl_stability": expl,
    }


def stable_merge_score(
    env_scores: Dict[str, float],
    stabilities: Dict[str, float],
    ranking_stability: float,
    w_perf: float = 0.55,
    w_cal: float = 0.15,
    w_expl: float = 0.15,
    w_rank: float = 0.15
) -> float:
    """
    Final STABLE-MERGE objective (single scalar).
    Higher is better.
    """
    avg_env_score = float(np.mean(list(env_scores.values()))) if env_scores else 0.0
    rank01 = 0.5 * (ranking_stability + 1.0)

    return float(
        w_perf * avg_env_score
        + w_cal * stabilities["cal_stability"]
        + w_expl * stabilities["expl_stability"]
        + w_rank * rank01
    )


# ----------------------------
# Optional: Stable ensemble merging (weights over candidate models)
# ----------------------------
def stable_weighted_ensemble(
    per_model_probs_by_env: Dict[str, Dict[str, np.ndarray]],
    per_env_y: Dict[str, np.ndarray],
    gamma_var: float = 2.0,
    l2: float = 1e-2,
) -> Dict[str, Any]:
    """
    Learn weights over models to minimize:
      mean_env logloss + gamma_var * var_env(logloss) + l2 * ||w||^2
    subject to w >= 0, sum w = 1

    Requires SciPy. If not available, returns None.
    """
    if not _HAS_SCIPY:
        return {"ok": False, "reason": "SciPy not installed, skipping ensemble optimization."}

    model_names = sorted(per_model_probs_by_env.keys())
    envs = sorted(next(iter(per_model_probs_by_env.values())).keys())

    # Stack probabilities: env -> [n_models, n_samples]
    probs_env = {}
    pbar_stack = tqdm(envs, desc="Ensemble: stack probs by env", leave=False)
    for e in pbar_stack:
        probs_env[e] = np.vstack([per_model_probs_by_env[m][e] for m in model_names])
        pbar_stack.set_postfix(env=e, shape=str(probs_env[e].shape), refresh=True)

    y_env = {e: per_env_y[e] for e in envs}

    eval_counter = {"n": 0}

    def objective(w):
        eval_counter["n"] += 1
        w = np.clip(w, 1e-12, 1.0)
        w = w / (w.sum() + 1e-12)

        losses = []
        for e in envs:
            p = np.dot(w, probs_env[e])
            p = np.clip(p, 1e-12, 1 - 1e-12)
            losses.append(log_loss(y_env[e], p))
        losses = np.array(losses, dtype=float)
        mean_loss = losses.mean()
        var_loss = losses.var()
        reg = l2 * float(np.sum(w * w))
        return mean_loss + gamma_var * var_loss + reg

    x0 = np.ones(len(model_names)) / len(model_names)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * len(model_names)

    # SciPy minimize ไม่มี callback ที่อัปเดตทุก eval แบบสวยมาก
    # แต่เราทำ progress แบบประมาณจาก maxiter
    maxiter = 500
    pbar_opt = tqdm(total=maxiter, desc="Ensemble: optimize weights (SLSQP)", leave=False)

    def callback(wk):
        # callback เรียกทุก iteration (ไม่ใช่ทุก objective call)
        pbar_opt.update(1)
        ww = np.clip(wk, 0.0, 1.0)
        ww = ww / (ww.sum() + 1e-12)
        # แสดง top-3 weights
        top = np.argsort(-ww)[:3]
        postfix = {model_names[i]: float(ww[i]) for i in top}
        postfix["obj_evals"] = eval_counter["n"]
        pbar_opt.set_postfix(postfix, refresh=True)

    res = minimize(
        objective, x0,
        bounds=bnds, constraints=cons, method="SLSQP",
        options={"maxiter": maxiter},
        callback=callback
    )
    pbar_opt.close()

    w = np.clip(res.x, 0.0, 1.0)
    w = w / (w.sum() + 1e-12)

    return {
        "ok": bool(res.success),
        "message": str(res.message),
        "weights": {m: float(w[i]) for i, m in enumerate(model_names)},
        "final_objective": float(res.fun),
        "objective_evals": int(eval_counter["n"]),
    }


# ----------------------------
# Main runner
# ----------------------------
@dataclass
class RunConfig:
    seed: int = 42
    n_splits: int = 5
    save_dir: str = "stable_merge_out"
    do_ensemble_merge: bool = True


def run_stable_merge(envs: List[EnvDataset], cfg: RunConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = logging.getLogger("STABLE_MERGE")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    zoo = model_zoo(cfg.seed)
    logger.info(f"Environments: {[e.name for e in envs]}")
    logger.info(f"Models: {list(zoo.keys())}")

    # Evaluate each model across envs (OOF)
    results: Dict[str, ModelResult] = {}
    per_model_probs_by_env: Dict[str, Dict[str, np.ndarray]] = {}
    per_env_y: Dict[str, np.ndarray] = {}

    # Progress: models
    pbar_models = tqdm(list(zoo.items()), desc="Models", leave=True)
    for model_name, model in pbar_models:
        pbar_models.set_postfix(model=model_name, stage="start", refresh=True)
        logger.info(f"Evaluating model: {model_name}")

        env_metrics: Dict[str, Metrics] = {}
        env_probs: Dict[str, np.ndarray] = {}
        env_y: Dict[str, np.ndarray] = {}
        env_expl: Dict[str, np.ndarray] = {}
        env_rank_score: Dict[str, float] = {}

        # Progress: envs per model
        pbar_envs = tqdm(envs, desc=f"  Envs for {model_name}", leave=False)
        for env in pbar_envs:
            pbar_envs.set_postfix(env=env.name, step="start", refresh=True)

            m, p, y, expl = oof_eval_env(env, model, cfg.n_splits, cfg.seed, pbar_env=pbar_envs)

            env_metrics[env.name] = m
            env_probs[env.name] = p
            env_y[env.name] = y
            env_expl[env.name] = expl
            env_rank_score[env.name] = env_score_for_ranking(m)

            per_env_y[env.name] = y  # safe to overwrite
            pbar_envs.set_postfix(
                env=env.name,
                auroc=f"{m.auroc:.3f}" if np.isfinite(m.auroc) else "nan",
                logloss=f"{m.logloss:.3f}",
                ece=f"{m.ece:.3f}",
                refresh=True
            )

        per_model_probs_by_env[model_name] = env_probs

        results[model_name] = ModelResult(
            model_name=model_name,
            env_metrics=env_metrics,
            env_oof_probs=env_probs,
            env_oof_y=env_y,
            env_expl=env_expl,
            ranking_score_by_env=env_rank_score,
            stability={},
            stable_merge_score=float("nan")
        )

        pbar_models.set_postfix(model=model_name, stage="done", refresh=True)

    # Build rankings per env (best->worst)
    ranks_by_env: Dict[str, List[str]] = {}
    pbar_rank = tqdm(envs, desc="Build ranks_by_env", leave=False)
    for env in pbar_rank:
        e = env.name
        ordered = sorted(zoo.keys(), key=lambda m: results[m].ranking_score_by_env[e], reverse=True)
        ranks_by_env[e] = ordered
        pbar_rank.set_postfix(env=e, top1=ordered[0] if ordered else "-", refresh=True)

    # Global ranking stability
    global_rank_stab = kendall_tau_rank_stability(ranks_by_env)

    # Final stable-merge score per model
    pbar_scores = tqdm(list(zoo.keys()), desc="Compute stable-merge scores", leave=False)
    for model_name in pbar_scores:
        pbar_scores.set_postfix(model=model_name, step="stabilities", refresh=True)
        stabs = aggregate_stabilities(results[model_name].env_metrics, results[model_name].env_expl, pbar=pbar_scores)

        pbar_scores.set_postfix(model=model_name, step="final_score", refresh=True)
        sm_score = stable_merge_score(
            env_scores=results[model_name].ranking_score_by_env,
            stabilities=stabs,
            ranking_stability=global_rank_stab
        )

        results[model_name].stability = stabs
        results[model_name].stable_merge_score = sm_score
        pbar_scores.set_postfix(model=model_name, score=f"{sm_score:.4f}", refresh=True)

    # Select best
    best_model = max(results.values(), key=lambda r: r.stable_merge_score)

    # Optional stable ensemble weights (merge)
    ensemble = None
    if cfg.do_ensemble_merge:
        ensemble = stable_weighted_ensemble(
            per_model_probs_by_env=per_model_probs_by_env,
            per_env_y=per_env_y,
            gamma_var=2.0,
            l2=1e-2
        )

    # Save report
    pbar_save = tqdm(total=3, desc="Save report", leave=False)

    report = {
        "config": asdict(cfg),
        "envs": [e.name for e in envs],
        "global_ranking_stability": float(global_rank_stab),
        "best_model": best_model.model_name,
        "best_score": float(best_model.stable_merge_score),
        "models": {
            m: {
                "stable_merge_score": float(results[m].stable_merge_score),
                "stability": results[m].stability,
                "env_metrics": {e: results[m].env_metrics[e].to_dict() for e in results[m].env_metrics},
                "env_rank_score": results[m].ranking_score_by_env,
            } for m in results
        },
        "ranks_by_env": ranks_by_env,
        "stable_ensemble": ensemble,
    }
    pbar_save.update(1)

    out_json = os.path.join(cfg.save_dir, f"stable_merge_report_{int(time.time())}.json")
    pbar_save.set_postfix(path=os.path.basename(out_json), refresh=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pbar_save.update(1)

    logger.info(f"Best model: {best_model.model_name} | score={best_model.stable_merge_score:.4f}")
    logger.info(f"Global ranking stability (Kendall tau avg): {global_rank_stab:.4f}")
    logger.info(f"Saved report: {out_json}")
    pbar_save.update(1)
    pbar_save.close()

    return report


# ----------------------------
# Convenience: run both experiments
# ----------------------------
def main():
    cfg = RunConfig(seed=42, n_splits=5, save_dir="stable_merge_out", do_ensemble_merge=True)

    # 1) MIMIC mortality: demo vs full
    mimic_envs = build_mimic_envs(
        demo_path="demo_analytic_dataset_mortality_all_admissions.csv",
        full_path="full_analytic_dataset_mortality_all_admissions.csv",
        target_col="label_mortality"
    )
    print("\n========================")
    print("EXPERIMENT A: MIMIC Mortality (Demo vs Full)")
    print("========================")
    run_stable_merge(mimic_envs, cfg)

    # 2) TG response: Synthetic vs NHANES
    tg_envs = build_tg_envs(
        nhanes_path="nhanes_rsce_dataset_clean.csv",
        synthetic_path="Synthetic_Dataset_1500_Patients_precise.csv",
        task="tg4h_high",
        quantile=0.75
    )
    print("\n========================")
    print("EXPERIMENT B: TG Response (Synthetic vs NHANES)")
    print("========================")
    run_stable_merge(tg_envs, cfg)


if __name__ == "__main__":
    main()