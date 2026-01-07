from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# -----------------------------
# Configuration (Colab defaults)
# -----------------------------

# ✅ ปรับ path ตรงนี้ให้ตรงกับไฟล์ของคุณใน Colab
# ตัวอย่าง: ถ้าอัปโหลดไฟล์ไว้ใน /content/ ก็แก้เป็น "/content/xxx.csv"
DEFAULT_PATHS = {
    "mimic_demo": "/content/demo_analytic_dataset_mortality_all_admissions.csv",
    "mimic_full": "/content/full_analytic_dataset_mortality_all_admissions.csv",
    "synthetic": "/content/Synthetic_Dataset_1500_Patients_precise.csv",
    "nhanes": "/content/nhanes_rsce_dataset_clean.csv",
}

RNG_SEED = 1337

# ✅ ค่าเริ่มต้นสำหรับ “กดรันแล้วได้ผลลัพธ์เลย”
# เปลี่ยน task ได้ 2 แบบ:
#  - mimic_mortality  (ต้องมี mimic_demo + mimic_full)
#  - tg_high_response (ต้องมี synthetic + nhanes)
COLAB_DEFAULT_ARGS = [
    "--task", "mimic_mortality",
    "--outdir", "runs/ext_mimic",
    # ตัวเลือกเพิ่มเติม:
    # "--rank-score", "auroc",
    # "--rank-margin", "0.0",
]


# -----------------------------
# Notebook/CLI helpers
# -----------------------------

def _in_notebook() -> bool:
    """Heuristic: True in Jupyter/Colab, False in normal python."""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell", "GoogleShell")
    except Exception:
        return False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _require_files(paths: Dict[str, str], keys: List[str]) -> bool:
    """Check if required files exist. If not, print instructions and return False."""
    missing = []
    for k in keys:
        fp = paths.get(k, "")
        if not fp or not Path(fp).exists():
            missing.append((k, fp))
    if missing:
        print("\n❌ ไม่พบไฟล์ที่จำเป็น:")
        for k, fp in missing:
            print(f"  - {k}: {fp}")
        print("\n✅ วิธีแก้ใน Colab:")
        print("  1) อัปโหลดไฟล์เข้า Colab (sidebar > Files > Upload) แล้วแก้ DEFAULT_PATHS ให้ชี้ไปที่ /content/ไฟล์.csv")
        print("  2) หรือถ้าไฟล์อยู่ใน Google Drive ให้ mount drive แล้วแก้ path ให้ตรง")
        print("\nตัวอย่าง:")
        print('  DEFAULT_PATHS["mimic_demo"] = "/content/demo_analytic_dataset_mortality_all_admissions.csv"')
        print('  DEFAULT_PATHS["mimic_full"] = "/content/full_analytic_dataset_mortality_all_admissions.csv"')
        return False
    return True


# -----------------------------
# Preprocess + Models
# -----------------------------

def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target_col])
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def _make_onehot() -> OneHotEncoder:
    """
    sklearn compatibility:
      - old: OneHotEncoder(..., sparse=True/False)
      - new: OneHotEncoder(..., sparse_output=True/False)
    We want sparse output (CSR) for efficiency.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocess(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # keep sparse-friendly
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def model_zoo(seed: int = RNG_SEED) -> Dict[str, object]:
    return {
        "LR": LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            n_jobs=None,
            class_weight="balanced",
            random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        ),
        "HGB": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=seed,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            max_iter=250,
            early_stopping=True,
            random_state=seed,
        ),
    }


# -----------------------------
# Metrics
# -----------------------------

@dataclass
class MetricBundle:
    auroc: float
    auprc: float
    logloss: float
    brier: float
    ece: float


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in tqdm(range(n_bins), desc="ECE bins", leave=False):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricBundle:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
    return MetricBundle(
        auroc=float(roc_auc_score(y_true, y_prob)),
        auprc=float(average_precision_score(y_true, y_prob)),
        logloss=float(log_loss(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=float(expected_calibration_error(y_true, y_prob, n_bins=15)),
    )


# -----------------------------
# Ranking stability
# -----------------------------

@dataclass
class RankingStability:
    ref_env: str
    score_name: str
    margin: float
    flip_rate: float
    preference_consistency: float
    spearman_mean: float


def _rankdata(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values)
    order = np.argsort(vals)
    ranks = np.empty_like(vals, dtype=float)
    ranks[order] = np.arange(len(vals), dtype=float)

    sorted_vals = vals[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = float(np.mean(ranks[order[i:j + 1]]))
            ranks[order[i:j + 1]] = avg
        i = j + 1

    return ranks + 1.0


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def compute_ranking_stability(
    scores_by_env: Dict[str, Dict[str, float]],
    score_name: str = "auroc",
    ref_env: Optional[str] = None,
    margin: float = 0.0,
) -> RankingStability:
    if not scores_by_env:
        raise ValueError("scores_by_env is empty")

    envs = list(scores_by_env.keys())
    if ref_env is None:
        ref_env = envs[0]
    if ref_env not in scores_by_env:
        raise ValueError(f"ref_env={ref_env} not found")

    models = sorted(list(scores_by_env[ref_env].keys()))
    if len(models) < 2:
        raise ValueError("Need >=2 models for ranking stability")

    ref_scores = np.array([scores_by_env[ref_env][m] for m in models], dtype=float)

    edges: List[Tuple[int, int]] = []
    for i in tqdm(range(len(models)), desc="Build preference edges", leave=False):
        for j in range(len(models)):
            if i == j:
                continue
            if ref_scores[i] >= ref_scores[j] + margin:
                edges.append((i, j))

    if not edges:
        spears = []
        for e in tqdm(envs, desc="Spearman (no edges)", leave=False):
            spears.append(_spearman_corr(ref_scores, np.array([scores_by_env[e][m] for m in models])))
        return RankingStability(
            ref_env=ref_env,
            score_name=score_name,
            margin=margin,
            flip_rate=0.0,
            preference_consistency=1.0,
            spearman_mean=float(np.mean(spears) if spears else 1.0),
        )

    flip = 0
    total = 0
    spearmans: List[float] = []
    for env in tqdm(envs, desc="Ranking stability across envs"):
        if env == ref_env:
            continue
        env_scores = np.array([scores_by_env[env][m] for m in models], dtype=float)
        spearmans.append(_spearman_corr(ref_scores, env_scores))
        for (i, j) in edges:
            total += 1
            if env_scores[i] < env_scores[j]:
                flip += 1

    flip_rate = flip / total if total > 0 else 0.0
    return RankingStability(
        ref_env=ref_env,
        score_name=score_name,
        margin=margin,
        flip_rate=float(flip_rate),
        preference_consistency=float(1.0 - flip_rate),
        spearman_mean=float(np.mean(spearmans) if spearmans else 1.0),
    )


# -----------------------------
# Integrity-aware (constrained) selection
# -----------------------------

@dataclass
class SelectionOutcome:
    policy: str
    chosen_model: str
    chosen_score: float
    constraints_satisfied: bool
    details: Dict[str, float]


def constrained_selection(
    model_table: pd.DataFrame,
    primary: str = "auroc",
    constraint_cols: Tuple[str, str] = ("ece", "flip_rate"),
    thresholds: Tuple[float, float] = (0.05, 0.25),
    tie_breaker: Optional[str] = "auprc",
) -> SelectionOutcome:
    ece_col, flip_col = constraint_cols
    tau, kappa = thresholds

    df = model_table.copy()
    df = df.sort_values(by=[primary] + ([tie_breaker] if tie_breaker else []), ascending=False)

    feasible = df[(df[ece_col] <= tau) & (df[flip_col] <= kappa)]
    if len(feasible) > 0:
        row = feasible.iloc[0]
        return SelectionOutcome(
            policy=f"constrained(max {primary} s.t. {ece_col}<=tau,{flip_col}<=kappa)",
            chosen_model=str(row["model"]),
            chosen_score=float(row[primary]),
            constraints_satisfied=True,
            details={ece_col: float(row[ece_col]), flip_col: float(row[flip_col]), "tau": float(tau), "kappa": float(kappa)},
        )

    row = df.iloc[0]
    return SelectionOutcome(
        policy=f"constrained-fallback(max {primary}; infeasible)",
        chosen_model=str(row["model"]),
        chosen_score=float(row[primary]),
        constraints_satisfied=False,
        details={ece_col: float(row[ece_col]), flip_col: float(row[flip_col]), "tau": float(tau), "kappa": float(kappa)},
    )


def selection_baselines(model_table: pd.DataFrame, primary: str = "auroc") -> List[SelectionOutcome]:
    df = model_table.copy()
    out: List[SelectionOutcome] = []

    row = df.sort_values(primary, ascending=False).iloc[0]
    out.append(SelectionOutcome("select_by_mean_primary", str(row["model"]), float(row[primary]), True, {}))

    worst_col = f"worst_{primary}"
    if worst_col in df.columns:
        row = df.sort_values(worst_col, ascending=False).iloc[0]
        out.append(SelectionOutcome("select_by_worstcase_primary", str(row["model"]), float(row[worst_col]), True, {}))

    if "ece" in df.columns:
        row = df.sort_values("ece", ascending=True).iloc[0]
        out.append(SelectionOutcome("select_by_calibration_only(min ECE)", str(row["model"]), float(row[primary]), True, {"ece": float(row["ece"])}))

    if "flip_rate" in df.columns:
        row = df.sort_values("flip_rate", ascending=True).iloc[0]
        out.append(SelectionOutcome("select_by_rank_stability_only(min flip)", str(row["model"]), float(row[primary]), True, {"flip_rate": float(row["flip_rate"])}))

    return out


# -----------------------------
# Training + OOF predictions
# -----------------------------

def train_oof_predictions(
    df: pd.DataFrame,
    target_col: str,
    models: Dict[str, object],
    n_splits: int = 5,
    seed: int = RNG_SEED,
) -> Dict[str, np.ndarray]:
    y = df[target_col].astype(int).to_numpy()

    drop_cols = [c for c in df.columns if c.lower() in {"hadm_id", "subject_id", "seqn", "id"}]
    X_df = df.drop(columns=[target_col] + drop_cols)

    cat_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    preprocess = build_preprocess(cat_cols, num_cols)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds: Dict[str, np.ndarray] = {name: np.full(len(df), np.nan, dtype=float) for name in models}

    fold_iter = tqdm(list(cv.split(X_df, y)), desc="CV folds", total=n_splits)
    for fold, (tr, te) in enumerate(fold_iter):
        X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
        y_tr = y[tr]

        for name, base_model in tqdm(list(models.items()), desc=f"Fit models (fold {fold+1}/{n_splits})", leave=False):
            clf = Pipeline(steps=[("prep", preprocess), ("model", base_model)])
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            preds[name][te] = prob

    for name, p in tqdm(list(preds.items()), desc="Sanity check preds", leave=False):
        if np.isnan(p).any():
            raise RuntimeError(f"OOF preds contain NaN for {name}")

    return preds


# -----------------------------
# Task handling
# -----------------------------

def load_mimic_demo_full(demo_path: str, full_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    demo = pd.read_csv(demo_path)
    full = pd.read_csv(full_path)
    if "label_mortality" not in demo.columns or "label_mortality" not in full.columns:
        raise ValueError("Expected label_mortality in both MIMIC datasets")
    return demo, full


def derive_binary_label_from_quantile(
    df: pd.DataFrame,
    label_col: str,
    q: float,
    ref_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    if label_col not in df.columns:
        raise ValueError(f"label_col={label_col} not in columns")

    x = df[label_col].astype(float)
    x_no_nan = x.dropna()
    threshold = float(x_no_nan.quantile(q)) if ref_threshold is None else float(ref_threshold)

    out = df.copy()
    out["label"] = (out[label_col].astype(float) >= threshold).astype(int)
    out = out.loc[~out[label_col].isna()].reset_index(drop=True)
    return out, threshold


def resolve_column_name(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    variants = [name.lower(), name.upper(), name.capitalize(), name.title()]
    variants.append(name.replace("h", "H"))
    variants.append(name.replace("H", "h"))
    for v in tqdm(variants, desc=f"Resolve column '{name}'", leave=False):
        if v in df.columns:
            return v
    raise ValueError(f"Could not resolve column '{name}' in columns: {list(df.columns)[:25]}...")


# -----------------------------
# Reporting & plotting
# -----------------------------

def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_heatmap(matrix: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(0.6 * len(xlabels) + 3, 0.6 * len(ylabels) + 3))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_bar(values: Dict[str, float], title: str, ylabel: str, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(values.keys())
    vals = [values[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -----------------------------
# Main evaluation logic
# -----------------------------

def evaluate_envs(
    env_dfs: Dict[str, pd.DataFrame],
    target_col: str,
    preds_by_env: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    seed: int = RNG_SEED,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, MetricBundle]]]:
    models = model_zoo(seed)
    metrics_dict: Dict[str, Dict[str, MetricBundle]] = {}
    rows = []

    for env, df in tqdm(list(env_dfs.items()), desc="Evaluate environments"):
        if preds_by_env is None or env not in preds_by_env:
            preds = train_oof_predictions(df, target_col, models=models, seed=seed)
        else:
            preds = preds_by_env[env]

        y = df[target_col].astype(int).to_numpy()
        metrics_dict[env] = {}
        for mname, yprob in tqdm(list(preds.items()), desc=f"Metrics ({env})", leave=False):
            mb = compute_metrics(y, yprob)
            metrics_dict[env][mname] = mb
            rows.append(
                {
                    "env": env,
                    "model": mname,
                    "auroc": mb.auroc,
                    "auprc": mb.auprc,
                    "logloss": mb.logloss,
                    "brier": mb.brier,
                    "ece": mb.ece,
                }
            )

    df_long = pd.DataFrame(rows)
    return df_long, metrics_dict


def aggregate_across_envs(df_long: pd.DataFrame, primary: str = "auroc") -> pd.DataFrame:
    metrics = ["auroc", "auprc", "logloss", "brier", "ece"]
    agg = df_long.groupby("model")[metrics].agg(["mean", "min", "max"]).reset_index()

    new_cols = ["model"]
    for m in tqdm(metrics, desc="Flatten agg cols", leave=False):
        for stat in ["mean", "min", "max"]:
            new_cols.append(f"{m}_{stat}")
    agg.columns = new_cols

    out = pd.DataFrame({"model": agg["model"]})
    out[primary] = agg[f"{primary}_mean"]
    out[f"worst_{primary}"] = agg[f"{primary}_min"]
    out["ece"] = agg["ece_mean"]
    out["auprc"] = agg["auprc_mean"]
    out["logloss"] = agg["logloss_mean"]
    return out


def build_scores_by_env(metrics_dict: Dict[str, Dict[str, MetricBundle]], score: str) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for env, md in tqdm(list(metrics_dict.items()), desc=f"Build scores_by_env[{score}]", leave=False):
        scores[env] = {}
        for model, mb in md.items():
            scores[env][model] = float(getattr(mb, score))
    return scores


def pairwise_flip_heatmap(scores_by_env: Dict[str, Dict[str, float]], ref_env: str, margin: float = 0.0) -> Tuple[np.ndarray, List[str]]:
    envs = list(scores_by_env.keys())
    models = sorted(list(scores_by_env[ref_env].keys()))
    ref = np.array([scores_by_env[ref_env][m] for m in models])

    edges = []
    for i in tqdm(range(len(models)), desc="Heatmap: build ref edges", leave=False):
        for j in range(len(models)):
            if i == j:
                continue
            if ref[i] >= ref[j] + margin:
                edges.append((i, j))

    heat = np.zeros((len(models), len(models)), dtype=float)
    denom = max(1, len(envs) - 1)

    for env in tqdm(envs, desc="Heatmap: count flips across envs"):
        if env == ref_env:
            continue
        s = np.array([scores_by_env[env][m] for m in models])
        for (i, j) in edges:
            if s[i] < s[j]:
                heat[i, j] += 1.0

    heat = heat / denom
    return heat, models


def model_flip_involvement_from_heatmap(
    scores_by_env: Dict[str, Dict[str, float]],
    ref_env: str,
    heat: np.ndarray,
    models: List[str],
    margin: float = 0.0,
) -> Dict[str, float]:
    ref = np.array([scores_by_env[ref_env][m] for m in models])
    involvement: Dict[str, float] = {}
    for i, m in tqdm(list(enumerate(models)), desc="Per-model flip involvement"):
        outgoing = [j for j in range(len(models)) if i != j and ref[i] >= ref[j] + margin]
        involvement[m] = 0.0 if not outgoing else float(np.mean(heat[i, outgoing]))
    return involvement


# -----------------------------
# CLI (Notebook-safe)
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> Optional[argparse.Namespace]:
    """
    CLI:
      python stable_merge_icml_extension.py --task ... --outdir ...
    Notebook/Colab:
      main(["--task","mimic_mortality","--outdir","runs/ext_mimic"])
    """
    p = argparse.ArgumentParser(add_help=True)

    p.add_argument("--task", type=str, choices=["mimic_mortality", "tg_high_response"], default=None)
    p.add_argument("--outdir", type=str, default=None)

    p.add_argument("--preds-json", type=str, default=None)
    p.add_argument("--mimic-demo", type=str, default=DEFAULT_PATHS["mimic_demo"])
    p.add_argument("--mimic-full", type=str, default=DEFAULT_PATHS["mimic_full"])
    p.add_argument("--synthetic", type=str, default=DEFAULT_PATHS["synthetic"])
    p.add_argument("--nhanes", type=str, default=DEFAULT_PATHS["nhanes"])

    p.add_argument("--label-col", type=str, default="TG4h")
    p.add_argument("--positive-quantile", type=float, default=0.75)

    p.add_argument("--ref-env", type=str, default=None)
    p.add_argument("--rank-score", type=str, default="auroc", choices=["auroc", "auprc", "logloss"])
    p.add_argument("--rank-margin", type=float, default=0.0)

    p.add_argument("--tau-ece", type=float, default=0.05)
    p.add_argument("--kappa-flip", type=float, default=0.25)

    if argv is None:
        argv = sys.argv[1:]

    # In notebook: if no args, use defaults (one-click)
    if _in_notebook() and len(argv) == 0:
        argv = COLAB_DEFAULT_ARGS

    args, _unknown = p.parse_known_args(argv)

    # If still missing, show help and return None in notebook
    if args.task is None or args.outdir is None:
        if _in_notebook():
            print("⚠️  ต้องระบุ --task และ --outdir")
            p.print_help()
            return None
        p.error("--task and --outdir are required")

    return args


def load_preds_json(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    raw = json.loads(Path(path).read_text())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for env, models in tqdm(list(raw.items()), desc="Load preds-json envs"):
        out[env] = {}
        for m, npy_path in tqdm(list(models.items()), desc=f"Load npy ({env})", leave=False):
            out[env][m] = np.load(npy_path)
    return out


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args is None:
        return

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "reports")
    ensure_dir(outdir / "figures")

    preds_by_env = load_preds_json(args.preds_json) if args.preds_json else None

    # ---------- Task setup + required file checks ----------
    if args.task == "mimic_mortality":
        # if user didn't override, check DEFAULT_PATHS
        ok = _require_files(
            {"mimic_demo": args.mimic_demo, "mimic_full": args.mimic_full},
            ["mimic_demo", "mimic_full"],
        )
        if not ok:
            return

        demo, full = load_mimic_demo_full(args.mimic_demo, args.mimic_full)
        env_dfs = {"demo": demo, "full": full}
        target_col = "label_mortality"

    elif args.task == "tg_high_response":
        ok = _require_files(
            {"synthetic": args.synthetic, "nhanes": args.nhanes},
            ["synthetic", "nhanes"],
        )
        if not ok:
            return

        synth = pd.read_csv(args.synthetic)
        nh = pd.read_csv(args.nhanes)

        synth_label_col = resolve_column_name(synth, args.label_col)
        nh_label_col = resolve_column_name(nh, args.label_col)
        synth_l, thr = derive_binary_label_from_quantile(synth, label_col=synth_label_col, q=args.positive_quantile)
        nh_l, _ = derive_binary_label_from_quantile(nh, label_col=nh_label_col, q=args.positive_quantile, ref_threshold=thr)

        common = sorted(set(synth_l.columns).intersection(set(nh_l.columns)))
        if "label" not in common:
            common.append("label")

        synth_l = synth_l[common].copy()
        nh_l = nh_l[common].copy()

        env_dfs = {"synthetic": synth_l, "nhanes": nh_l}
        target_col = "label"

        save_json(
            {"label_col": args.label_col, "positive_quantile": args.positive_quantile, "threshold": thr},
            outdir / "reports" / "label_definition.json",
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # ---------- Per-environment evaluation ----------
    df_long, metrics_dict = evaluate_envs(env_dfs, target_col=target_col, preds_by_env=preds_by_env)
    save_df(df_long, outdir / "reports" / "per_env_metrics.csv")

    # ---------- Aggregate across envs ----------
    model_table = aggregate_across_envs(df_long, primary="auroc")

    # ---------- Ranking stability ----------
    rank_score = args.rank_score
    scores_by_env = build_scores_by_env(metrics_dict, score=rank_score)
    if rank_score == "logloss":
        # higher is better => use negative logloss as "score"
        scores_by_env = {e: {m: -v for m, v in d.items()} for e, d in scores_by_env.items()}

    ref_env = args.ref_env or list(scores_by_env.keys())[0]
    rs = compute_ranking_stability(scores_by_env, score_name=rank_score, ref_env=ref_env, margin=args.rank_margin)

    heat, models = pairwise_flip_heatmap(scores_by_env, ref_env=ref_env, margin=args.rank_margin)
    per_model_flip = model_flip_involvement_from_heatmap(
        scores_by_env=scores_by_env,
        ref_env=ref_env,
        heat=heat,
        models=models,
        margin=args.rank_margin,
    )
    model_table["flip_rate"] = model_table["model"].map(per_model_flip)
    save_df(model_table, outdir / "reports" / "model_table_aggregated.csv")

    save_json(
        {
            "ranking_stability": {
                "ref_env": rs.ref_env,
                "score": rs.score_name,
                "margin": rs.margin,
                "global_flip_rate": rs.flip_rate,
                "preference_consistency": rs.preference_consistency,
                "spearman_mean": rs.spearman_mean,
            }
        },
        outdir / "reports" / "ranking_stability_summary.json",
    )

    # ---------- Plots ----------
    plot_heatmap(
        heat,
        xlabels=models,
        ylabels=models,
        title=f"Pairwise preference flip fraction vs ref='{ref_env}' (score={rank_score}, margin={args.rank_margin})",
        outpath=outdir / "figures" / "pairwise_flip_heatmap.png",
    )
    plot_bar(
        {m: per_model_flip[m] for m in models},
        title="Model-specific flip involvement (lower = more stable)",
        ylabel="Mean flip fraction",
        outpath=outdir / "figures" / "model_flip_involvement.png",
    )

    # ---------- Selection baselines + constrained selection ----------
    baselines = selection_baselines(model_table, primary="auroc")
    constrained = constrained_selection(
        model_table,
        primary="auroc",
        thresholds=(args.tau_ece, args.kappa_flip),
        tie_breaker="auprc",
    )
    all_outcomes = baselines + [constrained]

    sel_rows = []
    for o in tqdm(all_outcomes, desc="Build selection outcomes"):
        r = {
            "policy": o.policy,
            "chosen_model": o.chosen_model,
            "chosen_score": o.chosen_score,
            "constraints_satisfied": o.constraints_satisfied,
        }
        r.update(o.details)
        sel_rows.append(r)
    df_sel = pd.DataFrame(sel_rows)
    save_df(df_sel, outdir / "reports" / "selection_outcomes.csv")

    evidence = model_table[["model", "auroc", "worst_auroc", "ece", "flip_rate", "auprc", "logloss"]].copy()
    evidence = evidence.sort_values(by=["auroc"], ascending=False)
    save_df(evidence, outdir / "reports" / "accuracy_vs_stability_table.csv")

    print("\n=== STABLE-MERGE EXTENSION SUMMARY ===")
    print(f"Task: {args.task}")
    print(f"Environments: {list(env_dfs.keys())}")
    print(f"Ranking stability (global): flip_rate={rs.flip_rate:.3f}, spearman_mean={rs.spearman_mean:.3f}")
    print("\nTop models by mean AUROC:")
    print(evidence.head(5).to_string(index=False))
    print("\nSelection outcomes:")
    print(df_sel.to_string(index=False))
    print(f"\nSaved to: {outdir.resolve()}")

    print("\n📁 Outputs:")
    print(f"  - Reports:  {outdir / 'reports'}")
    print(f"  - Figures:  {outdir / 'figures'}")


# -----------------------------
# Auto-run in Colab (one-click)
# -----------------------------
if __name__ == "__main__":
    # In notebook with no args, it will use COLAB_DEFAULT_ARGS
    main([
    "--task", "mimic_mortality",
    "--outdir", "runs/ext_mimic",
    "--mimic-demo", "/content/demo_analytic_dataset_mortality_all_admissions.csv",
    "--mimic-full", "/content/full_analytic_dataset_mortality_all_admissions.csv",
])