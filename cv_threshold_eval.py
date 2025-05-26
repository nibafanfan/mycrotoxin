#!/usr/bin/env python3
"""
cv_threshold_eval.py – cross-validated threshold search with bootstrap CI
---------------------------------------------------------------------------
• Stratified k-fold CV (k given by --k, default 5)
• Optional repeats (e.g. --repeat 10 gives 10×k = 50 test folds)
• Inside each training split:
      – sweep thresholds (0.00 … 1.00, step 0.02)
      – pick the one with highest F1 on the train fold
• Collect per-fold metrics on the held-out fold
• Return mean ± std for AUROC / AUPRC / F1 / Prec / Recall
• Compute the median of all fold-specific thresholds
• 2 000-× bootstrap of that median → 95 % CI
"""

import argparse, numpy as np, pandas as pd, sys
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold
)

GRID   = np.linspace(0, 1, 51)   # 0.00, 0.02, …, 1.00
BOOT_N = 2000                   # bootstrap samples
EP     = ["Carcinogenicity", "Mutagenicity", "Genotoxicity"]

def best_threshold(y, p):
    best_f1, best_t = -1, 0
    for t in GRID:
        f1 = f1_score(y, p >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def cv_endpoint(df, ep, id_col, k, repeat, seed=42):
    y_all = df[f"True_{ep}"].dropna()
    idx   = y_all.index
    p_all = df.loc[idx, f"Pred_{ep}"].values
    y_all = y_all.values.astype(int)

    splitter = (StratifiedKFold(k, shuffle=True, random_state=seed)
                if repeat == 1 else
                RepeatedStratifiedKFold(
                    n_splits=k, n_repeats=repeat, random_state=seed))

    rows, chosen = [], []
    for train, test in splitter.split(np.zeros(len(y_all)), y_all):
        thr = best_threshold(y_all[train], p_all[train])
        chosen.append(thr)

        yhat = p_all[test] >= thr
        rows.append(dict(
            AUROC  = roc_auc_score(y_all[test], p_all[test])
                     if len(np.unique(y_all[test])) > 1 else np.nan,
            AUPRC  = average_precision_score(y_all[test], p_all[test]),
            F1     = f1_score(y_all[test], yhat, zero_division=0),
            Prec   = precision_score(y_all[test], yhat, zero_division=0),
            Recall = recall_score(y_all[test], yhat, zero_division=0)
        ))

    m = pd.DataFrame(rows).mean().to_dict()
    s = pd.DataFrame(rows).std().to_dict()

    rng = np.random.default_rng(0)
    meds = [np.median(rng.choice(chosen, len(chosen), replace=True))
            for _ in range(BOOT_N)]
    ci = np.percentile(meds, [2.5, 50, 97.5])

    return m, s, chosen, ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",  default="chemprop_canonical.csv")
    ap.add_argument("--truth", default="experimental_labels.csv")
    ap.add_argument("--id",    default="cid")
    ap.add_argument("--k",     type=int, default=5,
                    help="number of CV folds (default 5)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the k-fold CV n times (default 1)")
    args = ap.parse_args()

    df = pd.merge(pd.read_csv(args.truth), pd.read_csv(args.pred), on=args.id)

    print(f"Evaluating endpoints: {EP}\n")

    res = []
    for ep in EP:
        m, s, chosen, ci = cv_endpoint(df, ep, args.id, args.k, args.repeat)
        res.append(dict(
            Endpoint = ep,
            N        = df[f"True_{ep}"].notna().sum(),
            AUROC    = f"{m['AUROC']:.3f} ± {s['AUROC']:.3f}",
            AUPRC    = f"{m['AUPRC']:.3f} ± {s['AUPRC']:.3f}",
            F1       = f"{m['F1']:.3f} ± {s['F1']:.3f}",
            Prec     = f"{m['Prec']:.3f} ± {s['Prec']:.3f}",
            Recall   = f"{m['Recall']:.3f} ± {s['Recall']:.3f}",
            Fold_thresholds = ", ".join(f"{t:.2f}" for t in chosen),
            Median_threshold = f"{ci[1]:.2f}",
            CI95 = f"[{ci[0]:.2f}, {ci[2]:.2f}]"
        ))

    print(pd.DataFrame(res).to_string(index=False))

if __name__ == "__main__":
    main()
