#!/usr/bin/env python3
"""
cv_threshold_eval.py  –  small-N friendly evaluation
----------------------------------------------------
• Performs stratified 5-fold CV on each endpoint that has labels.
• Inside each training fold:
      – sweeps thresholds (grid step 0.02)
      – picks the one with the highest F1 on the *training* fold
• Applies that threshold to the held-out fold → metrics
• Collects:
      – mean ± std   (AUROC, AUPRC, F1, precision, recall)
      – the five chosen thresholds  → median threshold
• Bootstraps that median threshold to give a 95 % CI.
"""

import argparse, pandas as pd, numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold

GRID = np.linspace(0, 1, 51)          # 0.00, 0.02, … 1.00
BOOT_N = 2000                         # bootstrap draws

def pick_best_threshold(y, p):
    best_f1, best_t = -1, 0
    for t in GRID:
        yhat = (p >= t)
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def cv_endpoint(df, ep, id_col, k=5, seed=42):
    y = df[f"True_{ep}"].dropna()
    idx = y.index
    p = df.loc[idx, f"Pred_{ep}"].values
    y = y.values.astype(int)

    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    rows, chosen = [], []
    for train, test in skf.split(np.zeros(len(y)), y):
        # pick threshold on train
        thr = pick_best_threshold(y[train], p[train])
        chosen.append(thr)
        # apply to test
        yhat = (p[test] >= thr)
        rows.append(dict(
            AUROC  = roc_auc_score(y[test], p[test]) if len(np.unique(y[test]))>1 else np.nan,
            AUPRC  = average_precision_score(y[test], p[test]),
            F1     = f1_score(y[test], yhat, zero_division=0),
            Prec   = precision_score(y[test], yhat, zero_division=0),
            Recall = recall_score(y[test], yhat, zero_division=0)
        ))
    m = pd.DataFrame(rows).mean().to_dict()
    s = pd.DataFrame(rows).std().to_dict()
    # bootstrap the median of chosen thresholds
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
    a = ap.parse_args()

    df = pd.merge(pd.read_csv(a.truth), pd.read_csv(a.pred), on=a.id)
    endpoints = [c[len("True_"):] for c in df.columns if c.startswith("True_")]

    print(f"Evaluating endpoints: {endpoints}\n")

    res = []
    for ep in endpoints:
        m, s, chosen, ci = cv_endpoint(df, ep, a.id)
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
import pandas as pd

# 1. load predictions
pred = pd.read_csv("chemprop_canonical.csv")

# 2. assign flags using the CV-derived cut-offs
pred["Flag_Carc"] = (pred["Pred_Carcinogenicity"] >= 0.02).astype(int)
pred["Flag_Muta"] = (pred["Pred_Mutagenicity"]    >= 0.08).astype(int)

# Genotoxicity: 3-tier label
pred["Geno_Tier"] = pd.cut(
    pred["Pred_Genotoxicity"],
    bins=[-1, 0.05, 0.70, 1.01],
    labels=["negative", "borderline", "positive"]
)

pred.to_csv("chemprop_screened.csv", index=False)
print("✓ screening flags written → chemprop_screened.csv")
