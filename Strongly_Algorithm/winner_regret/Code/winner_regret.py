# winner_regret_from_reports.py
import argparse, json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="stable_merge_report_*.json")
    ap.add_argument("--out", default="winner_regret.csv")
    args = ap.parse_args()

    with open(args.report, "r") as f:
        rep = json.load(f)

    envs = rep["envs"]
    models = rep["models"]

    # Best-by-env using AUROC
    best_auc = {}
    best_model = {}
    for e in envs:
        best_auc[e] = -1.0
        best_model[e] = None
        for m, info in models.items():
            auc = info["env_metrics"][e]["auroc"]
            if auc > best_auc[e]:
                best_auc[e] = auc
                best_model[e] = m

    # Selected models:
    # 1) baseline winner picked in env0
    sel_env0 = best_model[envs[0]]
    # 2) baseline winner picked in env1
    sel_env1 = best_model[envs[1]]
    # 3) stable-merge winner (as stored)
    sel_sm = rep.get("best_model")

    selectors = [
        ("BaselineWinner@" + envs[0], sel_env0),
        ("BaselineWinner@" + envs[1], sel_env1),
        ("StableMergeWinner", sel_sm),
    ]

    rows = []
    for target in envs:
        for name, sel in selectors:
            auc_sel = models[sel]["env_metrics"][target]["auroc"]
            regret = best_auc[target] - auc_sel
            rows.append({
                "target_env": target,
                "selector": name,
                "selected_model": sel,
                "auc_in_target": auc_sel,
                "best_auc_in_target": best_auc[target],
                "winner_regret": regret
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df)

if __name__ == "__main__":
    main()
