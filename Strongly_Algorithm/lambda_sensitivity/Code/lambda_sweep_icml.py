# lambda_sweep_icml.py
# Weight-sweep for STABLE-MERGE (ICML version)
# - Re-runs stable_merge_icml.run_stable_merge with different (w_perf, w_cal, w_expl, w_rank)
# - Main focus: MIMIC (Demo vs Full). TG (NHANES/Synthetic) optional.
#
# Usage (Colab):
#   python lambda_sweep_icml.py --mimic_only \
#     --demo /content/demo_analytic_dataset_mortality_all_admissions.csv \
#     --full /content/full_analytic_dataset_mortality_all_admissions.csv \
#     --outdir /content/out_lambda_icml \
#     --grid 0,0.05,0.1,0.15,0.2,0.25
#
# Optional run TG too:
#   python lambda_sweep_icml.py --run_tg \
#     --nhanes /content/nhanes_rsce_dataset_clean.csv \
#     --synthetic /content/Synthetic_Dataset_1500_Patients_precise.csv \
#     ... (plus mimic args)

import argparse
import os
import json
import time
from copy import deepcopy
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORTANT: stable_merge_icml.py must be in the same folder OR in PYTHONPATH
import stable_merge_icml as sm


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def compute_winner_regret_from_report(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Winner regret on AUROC using report['models'][m]['env_metrics'][env]['auroc'].
    regret(target_env, selector) = best_auc_in_target - auc(selected_model in target_env)
    """
    envs = report["envs"]
    models = report["models"]

    # Best-by-env
    best_auc = {}
    best_model = {}
    for e in envs:
        best_auc[e] = -1.0
        best_model[e] = None
        for m, info in models.items():
            auc = float(info["env_metrics"][e]["auroc"])
            if auc > best_auc[e]:
                best_auc[e] = auc
                best_model[e] = m

    selectors = [
        (f"BaselineWinner@{envs[0]}", best_model[envs[0]]),
        (f"BaselineWinner@{envs[1]}", best_model[envs[1]]),
        ("StableMergeWinner", report["best_model"]),
    ]

    rows = []
    for target in envs:
        for sel_name, sel_model in selectors:
            auc_sel = float(models[sel_model]["env_metrics"][target]["auroc"])
            rows.append({
                "target_env": target,
                "selector": sel_name,
                "selected_model": sel_model,
                "auc_in_target": auc_sel,
                "best_auc_in_target": best_auc[target],
                "winner_regret": float(best_auc[target] - auc_sel),
            })
    return pd.DataFrame(rows)


def patch_stable_merge_score(w_perf: float, w_cal: float, w_expl: float, w_rank: float):
    """
    Monkey-patch sm.stable_merge_score so that run_stable_merge uses these weights.
    """
    original = sm.stable_merge_score

    def wrapped(env_scores, stabilities, ranking_stability, **kwargs):
        return original(
            env_scores=env_scores,
            stabilities=stabilities,
            ranking_stability=ranking_stability,
            w_perf=w_perf,
            w_cal=w_cal,
            w_expl=w_expl,
            w_rank=w_rank
        )

    sm.stable_merge_score = wrapped
    return original  # return original for restore


def run_one_experiment(
    envs: List[sm.EnvDataset],
    cfg: sm.RunConfig,
    weights: Tuple[float, float, float, float],
    tag: str,
    outdir: str,
    do_ensemble_merge: bool,
) -> Dict[str, Any]:
    w_perf, w_cal, w_expl, w_rank = weights

    # patch stable_merge_score
    original_fn = patch_stable_merge_score(w_perf, w_cal, w_expl, w_rank)

    # configure output
    cfg_local = deepcopy(cfg)
    cfg_local.save_dir = outdir
    cfg_local.do_ensemble_merge = bool(do_ensemble_merge)

    # run
    report = sm.run_stable_merge(envs, cfg_local)

    # restore
    sm.stable_merge_score = original_fn

    # attach metadata
    report_meta = {
        "tag": tag,
        "weights": {"w_perf": w_perf, "w_cal": w_cal, "w_expl": w_expl, "w_rank": w_rank},
        "timestamp": int(time.time()),
    }
    with open(os.path.join(outdir, "lambda_sweep_meta.json"), "w", encoding="utf-8") as f:
        json.dump(report_meta, f, indent=2)

    # compute regret
    regret_df = compute_winner_regret_from_report(report)
    regret_df.to_csv(os.path.join(outdir, "winner_regret.csv"), index=False)

    return {
        "tag": tag,
        "w_perf": w_perf,
        "w_cal": w_cal,
        "w_expl": w_expl,
        "w_rank": w_rank,
        "best_model": report["best_model"],
        "best_score": float(report["best_score"]),
        "global_ranking_stability": float(report["global_ranking_stability"]),
        # convenient summary regrets
        "regret_demo": float(regret_df[(regret_df["target_env"] == report["envs"][0]) &
                                       (regret_df["selector"] == "StableMergeWinner")]["winner_regret"].iloc[0]),
        "regret_full": float(regret_df[(regret_df["target_env"] == report["envs"][1]) &
                                       (regret_df["selector"] == "StableMergeWinner")]["winner_regret"].iloc[0]),
        "regret_mean": float(regret_df[regret_df["selector"] == "StableMergeWinner"]["winner_regret"].mean()),
    }


def parse_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    # data paths
    ap.add_argument("--demo", required=True, help="demo_analytic_dataset_mortality_all_admissions.csv")
    ap.add_argument("--full", required=True, help="full_analytic_dataset_mortality_all_admissions.csv")
    ap.add_argument("--nhanes", default=None, help="nhanes_rsce_dataset_clean.csv (optional)")
    ap.add_argument("--synthetic", default=None, help="Synthetic_Dataset_1500_Patients_precise.csv (optional)")

    # run toggles
    ap.add_argument("--mimic_only", action="store_true", help="Run only MIMIC sweep (recommended).")
    ap.add_argument("--run_tg", action="store_true", help="Also run TG sweep (NHANES vs SYNTHETIC).")

    # sweep config
    ap.add_argument("--outdir", default="out_lambda_icml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--do_ensemble_merge", action="store_true",
                    help="If set, runs SciPy weight-ensemble too (slower). Default off for sweep.")

    # weights: fix cal/expl, sweep rank
    ap.add_argument("--w_cal", type=float, default=0.15)
    ap.add_argument("--w_expl", type=float, default=0.15)
    ap.add_argument("--grid", default="0,0.05,0.1,0.15,0.2,0.25",
                    help="Values for w_rank. w_perf = 1 - (w_cal+w_expl+w_rank).")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Build envs
    mimic_envs = sm.build_mimic_envs(args.demo, args.full, target_col="label_mortality")
    tg_envs = None
    if args.run_tg:
        if not args.nhanes or not args.synthetic:
            raise ValueError("--run_tg requires --nhanes and --synthetic paths.")
        tg_envs = sm.build_tg_envs(
            nhanes_path=args.nhanes,
            synthetic_path=args.synthetic,
            task="tg4h_high",
            quantile=0.75
        )

    cfg = sm.RunConfig(seed=args.seed, n_splits=args.splits, save_dir="tmp", do_ensemble_merge=False)

    grid = parse_grid(args.grid)

    rows_mimic = []
    rows_tg = []

    # Sweep
    for w_rank in grid:
        w_cal = float(args.w_cal)
        w_expl = float(args.w_expl)
        w_perf = 1.0 - (w_cal + w_expl + w_rank)

        if w_perf < 0:
            print(f"[SKIP] w_rank={w_rank:.3f} makes w_perf<0 (w_perf={w_perf:.3f}). Reduce grid or w_cal/w_expl.")
            continue

        weights = (w_perf, w_cal, w_expl, w_rank)

        # ---- MIMIC sweep (main) ----
        if (not args.run_tg) or args.mimic_only or True:
            tag = f"MIMIC_wrank_{w_rank:.3f}"
            od = os.path.join(args.outdir, tag)
            ensure_dir(od)
            print(f"\n=== Running {tag} weights={weights} ===")
            r = run_one_experiment(
                envs=mimic_envs,
                cfg=cfg,
                weights=weights,
                tag=tag,
                outdir=od,
                do_ensemble_merge=args.do_ensemble_merge
            )
            rows_mimic.append(r)

        # ---- TG sweep (optional support) ----
        if args.run_tg and tg_envs is not None:
            tag = f"TG_wrank_{w_rank:.3f}"
            od = os.path.join(args.outdir, tag)
            ensure_dir(od)
            print(f"\n=== Running {tag} weights={weights} ===")
            r = run_one_experiment(
                envs=tg_envs,
                cfg=cfg,
                weights=weights,
                tag=tag,
                outdir=od,
                do_ensemble_merge=args.do_ensemble_merge
            )
            rows_tg.append(r)

    # Save summaries
    if rows_mimic:
        dfm = pd.DataFrame(rows_mimic).sort_values("w_rank")
        dfm.to_csv(os.path.join(args.outdir, "lambda_sweep_summary_MIMIC.csv"), index=False)

        # plots
        plt.figure()
        plt.plot(dfm["w_rank"], dfm["best_score"], marker="o")
        plt.xlabel("w_rank")
        plt.ylabel("best STABLE-MERGE score")
        plt.title("MIMIC: Best score vs w_rank")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mimic_bestscore_vs_wrank.png"), dpi=200)

        plt.figure()
        plt.plot(dfm["w_rank"], dfm["regret_mean"], marker="o")
        plt.xlabel("w_rank")
        plt.ylabel("StableMerge winner regret (mean across envs)")
        plt.title("MIMIC: Winner regret vs w_rank")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mimic_regret_vs_wrank.png"), dpi=200)

        plt.figure()
        plt.plot(dfm["w_rank"], dfm["global_ranking_stability"], marker="o")
        plt.xlabel("w_rank")
        plt.ylabel("Global ranking stability (Kendall tau avg)")
        plt.title("MIMIC: Global ranking stability vs w_rank")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mimic_tau_vs_wrank.png"), dpi=200)

        print("\n=== Saved MIMIC sweep summary ===")
        print(dfm[["w_rank", "w_perf", "best_model", "best_score", "global_ranking_stability",
                   "regret_demo", "regret_full", "regret_mean"]])

    if rows_tg:
        dft = pd.DataFrame(rows_tg).sort_values("w_rank")
        dft.to_csv(os.path.join(args.outdir, "lambda_sweep_summary_TG.csv"), index=False)
        print("\n=== Saved TG sweep summary ===")
        print(dft[["w_rank", "w_perf", "best_model", "best_score", "global_ranking_stability",
                   "regret_demo", "regret_full", "regret_mean"]])

    print("\nDONE.")


if __name__ == "__main__":
    main()

