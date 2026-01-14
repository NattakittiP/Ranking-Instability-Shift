
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost (install if missing)
try:
    from xgboost import XGBClassifier
except Exception:
    !pip -q install xgboost
    from xgboost import XGBClassifier


# ---------------------------
# Config
# ---------------------------
# ปรับ path ให้ตรงกับ environment ของคุณ:
DEMO_PATH = "/content/demo_analytic_dataset_mortality_all_admissions.csv"
FULL_PATH = "/content/full_analytic_dataset_mortality_all_admissions.csv"

TARGET_COL = "label_mortality"
GROUP_COL = "subject_id"   # patient-level grouping (critical)

N_SPLITS = 5
RANDOM_STATE = 42

# deltas to report (as suggested by your advisor)
DELTA_GRID = [0.0, 0.001, 0.002, 0.005]

# paired bootstrap config
N_BOOT = 4000             # increase if you want tighter CI (but slower)
CI_ALPHA = 0.05           # 95% CI
BOOT_SEED = 123

OUTDIR = "flip_tie_sensitivity_outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------
# Models (match your paper candidate set)
# ---------------------------
def build_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "LogReg_L2": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            n_jobs=None,
            random_state=random_state
        ),
        "RF_500": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            class_weight=None
        ),
        "ExtraTrees_800": ExtraTreesClassifier(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1
        ),
        "MLP_256x128": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=random_state
        ),
        "XGB_600": XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss"
        )
    }
    return models


# Deterministic tie-break order (simpler → more complex)
MODEL_ORDER = ["LogReg_L2", "RF_500", "ExtraTrees_800", "MLP_256x128", "XGB_600"]
MODEL_RANK = {m: i for i, m in enumerate(MODEL_ORDER)}


# ---------------------------
# Utilities
# ---------------------------
def load_and_align(
    demo_path: str,
    full_path: str,
    target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    demo = pd.read_csv(demo_path)
    full = pd.read_csv(full_path)

    if target not in demo.columns or target not in full.columns:
        raise ValueError(f"Missing target column: {target}")

    # Use common columns only (guarantee identical feature schema across env)
    common_cols = sorted(list(set(demo.columns) & set(full.columns)))

    if GROUP_COL not in common_cols:
        raise ValueError(f"GROUP_COL={GROUP_COL} not found in BOTH datasets.")

    # Restrict to common schema
    demo = demo[common_cols].copy()
    full = full[common_cols].copy()

    # Drop rows missing group/target
    demo = demo.dropna(subset=[GROUP_COL, target]).reset_index(drop=True)
    full = full.dropna(subset=[GROUP_COL, target]).reset_index(drop=True)

    # Feature candidates (exclude target + group + obvious IDs)
    feat_cols = [c for c in common_cols if c not in [target, GROUP_COL, "hadm_id"]]

    # Split numeric / categorical by dtype (and keep intersection across envs)
    num_demo = demo[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    num_full = full[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_demo = demo[feat_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    cat_full = full[feat_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_cols = sorted(list(set(num_demo) & set(num_full)))
    cat_cols = sorted(list(set(cat_demo) & set(cat_full)))

    demo_out = demo[[GROUP_COL] + numeric_cols + cat_cols + [target]].copy()
    full_out = full[[GROUP_COL] + numeric_cols + cat_cols + [target]].copy()

    return demo_out, full_out, numeric_cols, cat_cols


def make_pipeline(
    model,
    numeric_cols: List[str],
    cat_cols: List[str],
    needs_scaling: bool
) -> Pipeline:
    # Numeric: median impute (+ optional scaling)
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        num_steps.append(("scaler", StandardScaler()))
    num_transform = Pipeline(steps=num_steps)

    # Categorical: most_frequent impute + onehot
    cat_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", num_transform, numeric_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_transform, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model)
    ])
    return pipe


def get_cv_splits(df: pd.DataFrame, target: str, group_col: str, n_splits: int, seed: int):
    y = df[target].astype(int).values
    groups = df[group_col].values
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(df, y, groups))


def oof_predictions(
    df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    target: str,
    group_col: str,
    models: Dict[str, object],
    splits
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    y = df[target].astype(int).values

    feature_cols = numeric_cols + cat_cols
    X = df[feature_cols].copy()

    # OOF arrays
    oof = {name: np.full(shape=(len(df),), fill_value=np.nan, dtype=float) for name in models.keys()}

    for fold, (tr, te) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_te = X.iloc[te]

        for name, base_model in models.items():
            needs_scaling = name in ["LogReg_L2", "MLP_256x128"]
            pipe = make_pipeline(base_model, numeric_cols, cat_cols, needs_scaling=needs_scaling)

            pipe.fit(X_tr, y_tr)

            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(X_te)[:, 1]
            else:
                # fallback (unlikely)
                s = pipe.decision_function(X_te)
                p = 1.0 / (1.0 + np.exp(-s))

            oof[name][te] = p

    for name in oof:
        if np.isnan(oof[name]).any():
            raise RuntimeError(f"NaNs in OOF predictions for {name}. Check splits/preprocessing.")
    return oof, y


def auc_from_oof(oof: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
    return {name: roc_auc_score(y, p) for name, p in oof.items()}


def paired_bootstrap_auc_diff(
    y: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int
) -> Dict[str, float]:
    """
    Paired bootstrap of AUROC difference: AUROC(p_i) - AUROC(p_j)
    Uses identical resampled indices for both models (paired).
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    diffs = np.empty(n_boot, dtype=float)
    idx = np.arange(n)

    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        if len(np.unique(y[samp])) < 2:
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            if len(pos) > 0 and len(neg) > 0:
                samp[0] = rng.choice(pos)
                samp[1] = rng.choice(neg)

        auc_i = roc_auc_score(y[samp], p_i[samp])
        auc_j = roc_auc_score(y[samp], p_j[samp])
        diffs[b] = auc_i - auc_j

    lo = np.quantile(diffs, alpha / 2.0)
    hi = np.quantile(diffs, 1.0 - alpha / 2.0)
    mean = float(np.mean(diffs))

    return {
        "diff_mean": mean,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "significant": bool((lo > 0.0) or (hi < 0.0)),  # CI excludes 0
    }


def preference_with_delta(
    auc_i: float,
    auc_j: float,
    delta: float,
    name_i: str,
    name_j: str,
    tie_mode: str = "deterministic"
) -> int:
    """
    Returns preference:
      +1 if i preferred, -1 if j preferred, 0 if tie (only if tie_mode='tie_as_tie')
    tie_mode:
      - "deterministic": if |diff|<=delta, break tie via fixed MODEL_ORDER (simpler wins)
      - "tie_as_tie": return 0 for ties
    """
    if auc_i > auc_j + delta:
        return +1
    if auc_j > auc_i + delta:
        return -1

    if tie_mode == "tie_as_tie":
        return 0

    # deterministic tie-break
    if MODEL_RANK[name_i] < MODEL_RANK[name_j]:
        return +1
    elif MODEL_RANK[name_j] < MODEL_RANK[name_i]:
        return -1
    else:
        return 0


def compute_flips_across_envs(
    auc_demo: Dict[str, float],
    auc_full: Dict[str, float],
    delta: float,
    tie_mode: str
) -> Dict[str, float]:
    """
    Flip summary between TWO environments (demo vs full)
    """
    names = list(auc_demo.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    flip = 0
    ties_demo = 0
    ties_full = 0
    effective = 0

    for a, b in pairs:
        p_d = preference_with_delta(auc_demo[a], auc_demo[b], delta, a, b, tie_mode=tie_mode)
        p_f = preference_with_delta(auc_full[a], auc_full[b], delta, a, b, tie_mode=tie_mode)

        if tie_mode == "tie_as_tie":
            if p_d == 0: ties_demo += 1
            if p_f == 0: ties_full += 1
            if p_d == 0 or p_f == 0:
                continue

        effective += 1
        if p_d != p_f:
            flip += 1

    total_pairs = len(pairs)
    denom = effective if tie_mode == "tie_as_tie" else total_pairs
    fliprate = flip / max(denom, 1)

    return {
        "delta": delta,
        "tie_mode": tie_mode,
        "total_pairs": total_pairs,
        "effective_pairs": denom,
        "flip_pairs": flip,
        "fliprate": fliprate,
        "ties_demo": ties_demo,
        "ties_full": ties_full,
    }


def significance_flip_summary(
    auc_demo: Dict[str, float],
    auc_full: Dict[str, float],
    boot_demo: Dict[Tuple[str, str], Dict[str, float]],
    boot_full: Dict[Tuple[str, str], Dict[str, float]],
    delta: float,
    tie_mode: str
) -> Dict[str, float]:
    """
    A flip is "significant" if:
      - preference differs across envs (under delta + tie mode)
      - AND in BOTH envs the AUROC difference CI excludes 0 (paired bootstrap)
    """
    names = list(auc_demo.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    flip_total = 0
    flip_sig = 0
    flip_amb = 0
    effective = 0

    for a, b in pairs:
        p_d = preference_with_delta(auc_demo[a], auc_demo[b], delta, a, b, tie_mode=tie_mode)
        p_f = preference_with_delta(auc_full[a], auc_full[b], delta, a, b, tie_mode=tie_mode)

        if tie_mode == "tie_as_tie" and (p_d == 0 or p_f == 0):
            continue

        effective += 1
        if p_d != p_f:
            flip_total += 1

            key = (a, b) if (a, b) in boot_demo else (b, a)
            sd = boot_demo[key]["significant"]
            sf = boot_full[key]["significant"]

            if sd and sf:
                flip_sig += 1
            else:
                flip_amb += 1

    return {
        "delta": delta,
        "tie_mode": tie_mode,
        "effective_pairs": effective,
        "flip_total": flip_total,
        "flip_sig": flip_sig,
        "flip_amb": flip_amb,
        "pct_sig_among_flips": (flip_sig / flip_total) if flip_total > 0 else 0.0,
        "fliprate_sig": (flip_sig / effective) if effective > 0 else 0.0,
        "fliprate_total": (flip_total / effective) if effective > 0 else 0.0
    }


# ---------------------------
# Main run
# ---------------------------
print("Loading + aligning datasets...")
demo_df, full_df, numeric_cols, cat_cols = load_and_align(DEMO_PATH, FULL_PATH, TARGET_COL)

print("Numeric feature cols:", len(numeric_cols))
print("Categorical feature cols:", len(cat_cols))
print("Demo shape:", demo_df.shape, "Full shape:", full_df.shape)

print("Building CV splits (grouped by subject_id)...")
demo_splits = get_cv_splits(demo_df, TARGET_COL, GROUP_COL, N_SPLITS, RANDOM_STATE)
full_splits = get_cv_splits(full_df, TARGET_COL, GROUP_COL, N_SPLITS, RANDOM_STATE)

print("Training models + generating OOF predictions (DEMO)...")
models = build_models(RANDOM_STATE)
demo_oof, y_demo = oof_predictions(demo_df, numeric_cols, cat_cols, TARGET_COL, GROUP_COL, models, demo_splits)

print("Training models + generating OOF predictions (FULL)...")
models = build_models(RANDOM_STATE)  # re-init
full_oof, y_full = oof_predictions(full_df, numeric_cols, cat_cols, TARGET_COL, GROUP_COL, models, full_splits)

print("Computing AUROC per model (OOF)...")
auc_demo = auc_from_oof(demo_oof, y_demo)
auc_full = auc_from_oof(full_oof, y_full)

auc_table = pd.DataFrame({
    "model": list(auc_demo.keys()),
    "AUROC_demo": [auc_demo[m] for m in auc_demo.keys()],
    "AUROC_full": [auc_full[m] for m in auc_demo.keys()],
})
auc_table["AUROC_mean"] = (auc_table["AUROC_demo"] + auc_table["AUROC_full"]) / 2.0
auc_table = auc_table.sort_values("AUROC_mean", ascending=False).reset_index(drop=True)
auc_table.to_csv(os.path.join(OUTDIR, "pairwise_auc_summary.csv"), index=False)
print("Saved:", os.path.join(OUTDIR, "pairwise_auc_summary.csv"))
display(auc_table)


# ---------------------------
# Paired bootstrap per model-pair, per environment
# ---------------------------
def compute_all_pairwise_boot(
    df_y: np.ndarray,
    oof: Dict[str, np.ndarray],
    n_boot: int,
    alpha: float,
    seed: int
) -> Dict[Tuple[str, str], Dict[str, float]]:
    names = list(oof.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = paired_bootstrap_auc_diff(
                df_y, oof[a], oof[b],
                n_boot=n_boot, alpha=alpha,
                seed=seed + 1000 * i + j
            )
            out[(a, b)] = res
    return out

print("Paired bootstrap (DEMO) ...")
boot_demo = compute_all_pairwise_boot(y_demo, demo_oof, N_BOOT, CI_ALPHA, BOOT_SEED)

print("Paired bootstrap (FULL) ...")
boot_full = compute_all_pairwise_boot(y_full, full_oof, N_BOOT, CI_ALPHA, BOOT_SEED)

def boot_to_df(boot: Dict[Tuple[str, str], Dict[str, float]], env_name: str) -> pd.DataFrame:
    rows = []
    for (a, b), r in boot.items():
        rows.append({
            "env": env_name,
            "model_i": a,
            "model_j": b,
            "diff_mean(AUROC_i - AUROC_j)": r["diff_mean"],
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "significant(CI_excludes_0)": r["significant"]
        })
    return pd.DataFrame(rows)

boot_demo_df = boot_to_df(boot_demo, "DEMO")
boot_full_df = boot_to_df(boot_full, "FULL")

boot_demo_df.to_csv(os.path.join(OUTDIR, "pairwise_diff_bootstrap_demo.csv"), index=False)
boot_full_df.to_csv(os.path.join(OUTDIR, "pairwise_diff_bootstrap_full.csv"), index=False)
print("Saved paired bootstrap tables.")


# ---------------------------
# Flip vs delta (two tie-handling modes)
# ---------------------------
flip_rows = []
for delta in DELTA_GRID:
    for tie_mode in ["deterministic", "tie_as_tie"]:
        s = compute_flips_across_envs(auc_demo, auc_full, delta=delta, tie_mode=tie_mode)
        flip_rows.append(s)

flip_df = pd.DataFrame(flip_rows)
flip_df.to_csv(os.path.join(OUTDIR, "flip_vs_delta.csv"), index=False)
print("Saved:", os.path.join(OUTDIR, "flip_vs_delta.csv"))
display(flip_df)

plt.figure()
for tie_mode in ["deterministic", "tie_as_tie"]:
    sub = flip_df[flip_df["tie_mode"] == tie_mode].sort_values("delta")
    plt.plot(sub["delta"], sub["fliprate"], marker="o", label=tie_mode)
plt.xlabel("delta (near-tie tolerance)")
plt.ylabel("FlipRate (demo vs full)")
plt.title("FlipRate vs delta (tie sensitivity)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "flip_vs_delta.png"), dpi=200)
plt.show()
print("Saved:", os.path.join(OUTDIR, "flip_vs_delta.png"))


# ---------------------------
# Significant flips vs ambiguous flips
# ---------------------------
sig_rows = []
for delta in DELTA_GRID:
    for tie_mode in ["deterministic", "tie_as_tie"]:
        r = significance_flip_summary(
            auc_demo, auc_full,
            boot_demo, boot_full,
            delta=delta, tie_mode=tie_mode
        )
        sig_rows.append(r)

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(os.path.join(OUTDIR, "flip_significance_summary.csv"), index=False)
print("Saved:", os.path.join(OUTDIR, "flip_significance_summary.csv"))
display(sig_df)

plt.figure()
for tie_mode in ["deterministic", "tie_as_tie"]:
    sub = sig_df[sig_df["tie_mode"] == tie_mode].sort_values("delta")
    plt.plot(sub["delta"], sub["fliprate_total"], marker="o", label=f"total flips ({tie_mode})")
    plt.plot(sub["delta"], sub["fliprate_sig"], marker="x", label=f"significant flips ({tie_mode})")
plt.xlabel("delta (near-tie tolerance)")
plt.ylabel("Flip rate")
plt.title("Total vs Significant FlipRate vs delta")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "flip_significance_summary.png"), dpi=200)
plt.show()
print("Saved:", os.path.join(OUTDIR, "flip_significance_summary.png"))

print("\nDONE. Outputs in:", OUTDIR)
print("Key files for Supplementary:")
print(" - flip_vs_delta.png + flip_vs_delta.csv")
print(" - flip_significance_summary.csv + flip_significance_summary.png")
print(" - pairwise_diff_bootstrap_demo.csv / full.csv (evidence)")
