# icml_tierA_env_stress_stats_guidance.py
# --------------------------------------
# Tier A (1-3) post-processing from per_env_metrics.csv
#
# Inputs: a CSV with columns including:
#   - model identifier (e.g., model, model_name, candidate)
#   - environment identifier (e.g., env, environment)
#   - AUROC per env (column name contains "auroc")
#   - ECE per env (column name contains "ece")  [optional but recommended]
#
# Outputs (in --outdir):
#   - flip_summary.csv
#   - flip_involvement.csv
#   - loo_env_out.csv
#   - reweight_sensitivity.csv
#   - feasibility_heatmap.csv + feasibility_heatmap.png
#   - mean_rank.csv
#   - kendall_tau.csv + kendall_tau.png
#   - kendall_distance.csv + kendall_distance.png
#
# Usage (Colab):
#   python icml_tierA_env_stress_stats_guidance.py \
#     --csv /content/MIMIC_per_env_metrics.csv \
#     --outdir /content/tierA_out_mimic \
#     --tau_grid 0.01,0.015,0.02,0.025,0.03 \
#     --kappa_grid 0.05,0.1,0.15,0.2,0.25,0.3 \
#     --tau_default 0.02 \
#     --kappa_default 0.2
#
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None


# ----------------------------
# Helpers: column detection
# ----------------------------
def _find_col(df: pd.DataFrame, candidates: List[str], contains: Optional[str] = None) -> Optional[str]:
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    # exact match candidates
    for cand in candidates:
        for c in cols:
            if lower[c] == cand.lower():
                return c

    # contains match
    if contains is not None:
        for c in cols:
            if contains.lower() in lower[c]:
                return c

    return None


def detect_schema(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    model_col = _find_col(df, ["model", "model_name", "candidate", "estimator", "clf", "learner"])
    env_col = _find_col(df, ["env", "environment", "eval_env", "evaluation_environment", "domain", "shift"])
    auroc_col = _find_col(df, ["auroc"], contains="auroc")
    ece_col = _find_col(df, ["ece", "expected_calibration_error"], contains="ece")

    missing = []
    if model_col is None:
        missing.append("model column (e.g., model/model_name/candidate)")
    if env_col is None:
        missing.append("environment column (e.g., env/environment)")
    if auroc_col is None:
        missing.append("AUROC column (name contains 'auroc')")

    if missing:
        raise ValueError(
            "Could not detect required columns: "
            + ", ".join(missing)
            + f"\nColumns found: {list(df.columns)}"
        )
    return model_col, env_col, auroc_col, ece_col


# ----------------------------
# Deterministic ordering & ranks
# ----------------------------
def deterministic_model_order(models: List[str]) -> List[str]:
    # fixed total order (lexicographic)
    return sorted(models)


def env_winner(auroc_col_vec: np.ndarray, model_order: List[str]) -> int:
    # returns index of winner under deterministic tie-break
    max_val = np.nanmax(auroc_col_vec)
    idxs = np.where(auroc_col_vec == max_val)[0]
    if len(idxs) == 1:
        return int(idxs[0])
    # tie-break by fixed model order: pick smallest order
    # idxs correspond to model_order ordering already
    return int(np.min(idxs))


def rank_models_desc(auroc_vec: np.ndarray, model_order: List[str]) -> np.ndarray:
    """
    Produce deterministic ranks 1..N (1 is best) for a single environment AUROC vector.
    Ties resolved deterministically using fixed model order.
    """
    n = len(auroc_vec)
    # sort by (-auroc, model_name)
    order = np.lexsort((np.array(model_order), -auroc_vec))
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    return ranks


# ----------------------------
# Flip metrics (existence-based)
# ----------------------------
def preference_sign(auroc_i, auroc_j, det_i_lt_j: bool, delta: float = 0.0) -> int:
    if np.isnan(auroc_i) or np.isnan(auroc_j):
        return 0
    diff = auroc_i - auroc_j

    # near-tie audit only when delta > 0
    if delta > 0.0 and abs(diff) <= delta:
        return 0

    if diff > 0:
        return +1
    if diff < 0:
        return -1

    # exact tie (diff == 0): deterministic tie-breaking
    return +1 if det_i_lt_j else -1


def compute_fliprate_and_flipinv(
    AU: pd.DataFrame,
    delta: float = 0.0,
) -> Tuple[float, pd.Series]:
    """
    AU: DataFrame index=models (fixed order), columns=environments, values=AUROC
    delta: near-tie threshold for tie-as-tie audit (default 0.0 => exact ties resolved deterministically)
    Returns:
      FlipRate_global (existence-based)
      FlipInv per model (fraction of opponent models with at least one flip)
    """
    models = list(AU.index)
    envs = list(AU.columns)
    n = len(models)
    if n < 2 or len(envs) < 2:
        return 0.0, pd.Series(0.0, index=models)

    # fixed order already AU.index; det_i_lt_j = True when i has smaller index than j
    flips_pair = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            seen_pos = False
            seen_neg = False
            for e in envs:
                s = preference_sign(AU.loc[models[i], e], AU.loc[models[j], e], det_i_lt_j=True, delta=delta)
                if s > 0:
                    seen_pos = True
                elif s < 0:
                    seen_neg = True
                if seen_pos and seen_neg:
                    flips_pair[i, j] = True
                    break

    num_pairs = n * (n - 1) / 2
    fliprate = float(np.sum(flips_pair)) / float(num_pairs)

    # FlipInv: per model, fraction of opponent models with a flip (existence-based)
    flipinv = np.zeros(n, dtype=float)
    for k in range(n):
        count = 0
        for j in range(n):
            if j == k:
                continue
            i, jj = (k, j) if k < j else (j, k)
            if flips_pair[i, jj]:
                count += 1
        flipinv[k] = count / float(n - 1)

    return fliprate, pd.Series(flipinv, index=models)


# ----------------------------
# Feasibility & selection
# ----------------------------
def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    # assumes weights sum to 1 over environments
    v = values.reindex(weights.index).to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mask = ~np.isnan(v)
    if not np.any(mask):
        return float("nan")
    w2 = w[mask]
    w2 = w2 / w2.sum()
    return float(np.sum(v[mask] * w2))


def compute_feasible_set(
    AU: pd.DataFrame,
    ECE: Optional[pd.DataFrame],
    tau_cal: float,
    kappa_rank: float,
    weights: Optional[pd.Series] = None,
    delta_for_flips: float = 0.0,
) -> Tuple[List[str], pd.Series, pd.Series, float, pd.Series]:
    """
    Returns:
      feasible_models (list)
      avg_U (Series over models)    (weighted if weights provided, else uniform over envs)
      avg_ECE (Series over models)  (weighted if weights provided, else uniform)  [nan if ECE is None]
      fliprate_global
      flipinv (Series)
    """
    envs = list(AU.columns)
    if weights is None:
        weights = pd.Series(1.0 / len(envs), index=envs)

    # Avg utility per model
    avg_U = AU.apply(lambda row: weighted_avg(row, weights), axis=1)

    # Avg ECE per model
    if ECE is not None:
        avg_ECE = ECE.apply(lambda row: weighted_avg(row, weights), axis=1)
    else:
        avg_ECE = pd.Series(np.nan, index=AU.index)

    # Flip metrics (existence-based) computed on AU (not weighted)
    fliprate, flipinv = compute_fliprate_and_flipinv(AU, delta=delta_for_flips)

    # Feasible set
    if ECE is None:
        # If no ECE provided, treat all as passing calibration (or you can force fail)
        cal_ok = pd.Series(True, index=AU.index)
    else:
        cal_ok = avg_ECE <= tau_cal

    rank_ok = flipinv <= kappa_rank
    feasible = list(AU.index[(cal_ok & rank_ok).to_numpy()])

    return feasible, avg_U, avg_ECE, fliprate, flipinv


def strict_select(feasible: List[str], avg_U: pd.Series) -> Optional[str]:
    if not feasible:
        return None
    sub = avg_U.loc[feasible]
    best = sub.max()
    # deterministic tie-break by index order
    winners = list(sub.index[sub == best])
    return winners[0]


def fallback_minimax_regret(
    AU: pd.DataFrame,
) -> Tuple[str, float]:
    """
    f^dagger = argmin_f max_{e,e'} ( AUROC(winner_in_e evaluated_on_e') - AUROC(f evaluated_on_e') )
    Winners in each e are determined deterministically from AU column.
    Returns: (fallback_model, worst_case_regret)
    """
    models = list(AU.index)
    envs = list(AU.columns)
    n = len(models)
    k = len(envs)

    # Precompute env-specific winner indices
    winner_idx = []
    for e in envs:
        col = AU[e].to_numpy(dtype=float)
        wi = env_winner(col, models)  # models already fixed order
        winner_idx.append(wi)

    # For each f, compute worst-case regret over e,e'
    worst_regret = np.zeros(n, dtype=float)
    for fi in range(n):
        wr = 0.0
        for ei in range(k):
            w = winner_idx[ei]
            for e2i in range(k):
                u_w = AU.iloc[w, e2i]
                u_f = AU.iloc[fi, e2i]
                if np.isnan(u_w) or np.isnan(u_f):
                    continue
                r = float(u_w - u_f)
                if r > wr:
                    wr = r
        worst_regret[fi] = wr

    min_wr = float(np.min(worst_regret))
    candidates = np.where(worst_regret == min_wr)[0]
    best_idx = int(candidates[0])  # deterministic
    return models[best_idx], min_wr


# ----------------------------
# LOO-E and reweighting
# ----------------------------
def make_weights_uniform(envs: List[str]) -> pd.Series:
    return pd.Series(1.0 / len(envs), index=envs)


def make_weights_heavy_one(envs: List[str], heavy_env: str, heavy: float = 0.7) -> pd.Series:
    if heavy_env not in envs:
        raise ValueError(f"heavy_env '{heavy_env}' not in environments.")
    if not (0.0 < heavy < 1.0):
        raise ValueError("heavy must be between 0 and 1.")
    rest = (1.0 - heavy) / (len(envs) - 1)
    w = pd.Series(rest, index=envs, dtype=float)
    w.loc[heavy_env] = heavy
    return w


# ----------------------------
# Stats baselines
# ----------------------------
def kendall_tau_and_distance(rank_a: np.ndarray, rank_b: np.ndarray) -> Tuple[float, int]:
    """
    rank arrays are permutations of 1..N (deterministic ranks).
    tau computed from concordant/discordant pairs.
    distance = number of discordant pairs (inversions between rankings).
    """
    n = len(rank_a)
    # Convert rank to order positions: smaller rank => earlier
    pos_a = np.argsort(rank_a)
    pos_b = np.argsort(rank_b)

    # Map model index -> position
    inv_a = np.empty(n, dtype=int); inv_b = np.empty(n, dtype=int)
    inv_a[pos_a] = np.arange(n)
    inv_b[pos_b] = np.arange(n)

    discord = 0
    concord = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = inv_a[i] - inv_a[j]
            db = inv_b[i] - inv_b[j]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    if total == 0:
        return float("nan"), 0
    tau = (concord - discord) / total
    return float(tau), int(discord)


def plot_heatmap(matrix: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, outpath: Path) -> None:
    plt.figure()
    plt.imshow(matrix, aspect="auto")
    plt.xticks(ticks=np.arange(len(xlabels)), labels=xlabels, rotation=90)
    plt.yticks(ticks=np.arange(len(ylabels)), labels=ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


@dataclass
class Config:
    csv: Path
    outdir: Path
    tau_grid: List[float]
    kappa_grid: List[float]
    tau_default: float
    kappa_default: float
    delta_flip: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to *_per_env_metrics.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--tau_grid", type=str, default="0.01,0.015,0.02,0.025,0.03",
                    help="Comma-separated tau_cal grid for feasibility heatmap")
    ap.add_argument("--kappa_grid", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3",
                    help="Comma-separated kappa_rank grid for feasibility heatmap")
    ap.add_argument("--tau_default", type=float, default=0.02, help="Default tau_cal for LOO-E & reweight")
    ap.add_argument("--kappa_default", type=float, default=0.2, help="Default kappa_rank for LOO-E & reweight")
    ap.add_argument("--delta_flip", type=float, default=0.0,
                    help="Delta for tie-as-tie flip audit (0.0 = exact ties resolved deterministically). "
                         "Keep 0.0 for main protocol.")
    args = ap.parse_args()

    cfg = Config(
        csv=Path(args.csv),
        outdir=Path(args.outdir),
        tau_grid=parse_float_list(args.tau_grid),
        kappa_grid=parse_float_list(args.kappa_grid),
        tau_default=float(args.tau_default),
        kappa_default=float(args.kappa_default),
        delta_flip=float(args.delta_flip),
    )

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.csv)
    model_col, env_col, auroc_col, ece_col = detect_schema(df)

    # Normalize identifiers
    df[model_col] = df[model_col].astype(str)
    df[env_col] = df[env_col].astype(str)

    models = deterministic_model_order(sorted(df[model_col].unique().tolist()))
    envs = sorted(df[env_col].unique().tolist())

    # AUROC matrix
    AU = df.pivot_table(index=model_col, columns=env_col, values=auroc_col, aggfunc="mean")
    AU = AU.reindex(index=models, columns=envs)

    # ECE matrix (optional)
    ECE = None
    if ece_col is not None:
        ECE = df.pivot_table(index=model_col, columns=env_col, values=ece_col, aggfunc="mean")
        ECE = ECE.reindex(index=models, columns=envs)

    # ------------------------
    # Global flip summary (full E)
    # ------------------------
    fliprate, flipinv = compute_fliprate_and_flipinv(AU, delta=cfg.delta_flip)
    summary = pd.DataFrame([{
        "N_models": len(models),
        "K_envs": len(envs),
        "FlipRate_global": fliprate,
        "delta_flip": cfg.delta_flip,
    }])
    summary.to_csv(cfg.outdir / "flip_summary.csv", index=False)
    flipinv.rename("FlipInv").to_csv(cfg.outdir / "flip_involvement.csv", index=True)

    # ------------------------
    # Feasibility heatmap (tau x kappa) under uniform weights
    # ------------------------
    if ECE is None:
        # Still produce heatmap, but calibration constraint is treated as pass
        pass

    heat = np.zeros((len(cfg.tau_grid), len(cfg.kappa_grid)), dtype=float)
    heat_count = np.zeros_like(heat)
    for i, tau in enumerate(cfg.tau_grid):
        for j, kappa in enumerate(cfg.kappa_grid):
            feasible, avg_U, avg_ECE, fr, fi = compute_feasible_set(
                AU, ECE, tau_cal=tau, kappa_rank=kappa,
                weights=make_weights_uniform(envs),
                delta_for_flips=cfg.delta_flip,
            )
            heat[i, j] = 1.0 if len(feasible) > 0 else 0.0
            heat_count[i, j] = len(feasible)

    # Save both feasible indicator and feasible count
    heat_df = pd.DataFrame(heat, index=[f"{x:g}" for x in cfg.tau_grid], columns=[f"{x:g}" for x in cfg.kappa_grid])
    heat_df.index.name = "tau_cal"
    heat_df.columns.name = "kappa_rank"
    heat_df.to_csv(cfg.outdir / "feasibility_heatmap.csv")

    heatc_df = pd.DataFrame(heat_count, index=[f"{x:g}" for x in cfg.tau_grid], columns=[f"{x:g}" for x in cfg.kappa_grid])
    heatc_df.index.name = "tau_cal"
    heatc_df.columns.name = "kappa_rank"
    heatc_df.to_csv(cfg.outdir / "feasibility_count_heatmap.csv")

    plot_heatmap(
        heat,
        xlabels=[f"{x:g}" for x in cfg.kappa_grid],
        ylabels=[f"{x:g}" for x in cfg.tau_grid],
        title="Feasibility (1=feasible, 0=infeasible)",
        outpath=cfg.outdir / "feasibility_heatmap.png",
    )
    plot_heatmap(
        heat_count,
        xlabels=[f"{x:g}" for x in cfg.kappa_grid],
        ylabels=[f"{x:g}" for x in cfg.tau_grid],
        title="Feasible set size |F|",
        outpath=cfg.outdir / "feasibility_count_heatmap.png",
    )

    # ------------------------
    # LOO-E stress test under default (tau_default, kappa_default)
    # ------------------------
    loo_rows = []
    for drop_env in envs:
        keep_envs = [e for e in envs if e != drop_env]
        AU_sub = AU[keep_envs]
        ECE_sub = ECE[keep_envs] if ECE is not None else None

        feasible, avg_U, avg_ECE, fr, fi = compute_feasible_set(
            AU_sub, ECE_sub,
            tau_cal=cfg.tau_default,
            kappa_rank=cfg.kappa_default,
            weights=make_weights_uniform(keep_envs),
            delta_for_flips=cfg.delta_flip,
        )
        strict = strict_select(feasible, avg_U)
        infeasible = (len(feasible) == 0)

        fb_model, fb_regret = fallback_minimax_regret(AU_sub)

        loo_rows.append({
            "env_dropped": drop_env,
            "K_envs_kept": len(keep_envs),
            "FlipRate_global": fr,
            "Feasible_set_size": len(feasible),
            "Strict_selected": strict if strict is not None else "",
            "Infeasible": bool(infeasible),
            "Fallback_model": fb_model,
            "Fallback_worst_regret": fb_regret,
        })

    pd.DataFrame(loo_rows).to_csv(cfg.outdir / "loo_env_out.csv", index=False)

    # ------------------------
    # Reweight sensitivity: uniform + heavy-one-env (0.7/0.3) for each env
    # ------------------------
    rw_rows = []
    # uniform baseline
    w_uni = make_weights_uniform(envs)
    feasible, avg_U, avg_ECE, fr, fi = compute_feasible_set(
        AU, ECE,
        tau_cal=cfg.tau_default, kappa_rank=cfg.kappa_default,
        weights=w_uni, delta_for_flips=cfg.delta_flip,
    )
    strict = strict_select(feasible, avg_U)
    fb_model, fb_regret = fallback_minimax_regret(AU)

    rw_rows.append({
        "scenario": "uniform",
        "weights": json.dumps(w_uni.to_dict()),
        "FlipRate_global": fr,
        "Feasible_set_size": len(feasible),
        "Strict_selected": strict if strict is not None else "",
        "Infeasible": bool(len(feasible) == 0),
        "Fallback_model": fb_model,
        "Fallback_worst_regret": fb_regret,
    })

    for heavy_env in envs:
        w = make_weights_heavy_one(envs, heavy_env=heavy_env, heavy=0.7)
        feasible, avg_U, avg_ECE, fr, fi = compute_feasible_set(
            AU, ECE,
            tau_cal=cfg.tau_default, kappa_rank=cfg.kappa_default,
            weights=w, delta_for_flips=cfg.delta_flip,
        )
        strict = strict_select(feasible, avg_U)
        fb_model, fb_regret = fallback_minimax_regret(AU)

        rw_rows.append({
            "scenario": f"heavy_{heavy_env}_0.7",
            "weights": json.dumps(w.to_dict()),
            "FlipRate_global": fr,
            "Feasible_set_size": len(feasible),
            "Strict_selected": strict if strict is not None else "",
            "Infeasible": bool(len(feasible) == 0),
            "Fallback_model": fb_model,
            "Fallback_worst_regret": fb_regret,
        })

    pd.DataFrame(rw_rows).to_csv(cfg.outdir / "reweight_sensitivity.csv", index=False)

    # ------------------------
    # Stats baselines: mean rank + Friedman; Kendall tau/distance across environments
    # ------------------------
    rank_mat = np.zeros((len(models), len(envs)), dtype=int)
    for c, e in enumerate(envs):
        rank_mat[:, c] = rank_models_desc(AU[e].to_numpy(dtype=float), models)

    mean_rank = rank_mat.mean(axis=1)
    mean_rank_df = pd.DataFrame({
        "model": models,
        "mean_rank": mean_rank,
        "mean_AUROC": AU.mean(axis=1).to_numpy(dtype=float),
    }).sort_values(["mean_rank", "model"], ascending=[True, True])
    mean_rank_df.to_csv(cfg.outdir / "mean_rank.csv", index=False)

    # Friedman test (if available)
    friedman_out = {}
    if stats is not None and len(envs) >= 3:
        # scipy expects each "treatment" across blocks; here blocks=envs, treatments=models
        # Use ranks per env as "performance" for each model across envs
        try:
            args_for_f = [rank_mat[i, :] for i in range(rank_mat.shape[0])]
            stat, p = stats.friedmanchisquare(*args_for_f)
            friedman_out = {"friedman_stat": float(stat), "friedman_p": float(p)}
        except Exception:
            friedman_out = {}
    pd.DataFrame([friedman_out]).to_csv(cfg.outdir / "friedman_test.csv", index=False)

    # Kendall tau + distance matrices between env rankings
    K = len(envs)
    tau_m = np.full((K, K), np.nan, dtype=float)
    dist_m = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            if i == j:
                tau_m[i, j] = 1.0
                dist_m[i, j] = 0
            elif i < j:
                tau, dist = kendall_tau_and_distance(rank_mat[:, i], rank_mat[:, j])
                tau_m[i, j] = tau
                tau_m[j, i] = tau
                dist_m[i, j] = dist
                dist_m[j, i] = dist

    pd.DataFrame(tau_m, index=envs, columns=envs).to_csv(cfg.outdir / "kendall_tau.csv")
    pd.DataFrame(dist_m, index=envs, columns=envs).to_csv(cfg.outdir / "kendall_distance.csv")

    plot_heatmap(
        tau_m,
        xlabels=envs, ylabels=envs,
        title="Kendall tau between environment rankings",
        outpath=cfg.outdir / "kendall_tau.png",
    )
    plot_heatmap(
        dist_m.astype(float),
        xlabels=envs, ylabels=envs,
        title="Kendall distance (discordant pairs)",
        outpath=cfg.outdir / "kendall_distance.png",
    )

    # Done
    print(f"[OK] Wrote outputs to: {cfg.outdir.resolve()}")


if __name__ == "__main__":
    main()
