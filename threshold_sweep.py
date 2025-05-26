#!/usr/bin/env python3
"""
threshold_sweep.py – choose an operating threshold

• Reads chemprop_canonical.csv  (Probabilities)
• Reads experimental_labels.csv (Truth)
• Sweeps thresholds 0.00 … 1.00 (step 0.05)
• Prints a neat table and highlights the threshold with the
  highest F1-score for every endpoint.

Optional:
    --step 0.01    # finer resolution
    --metric f1    # or 'youden' to maximise (TPR-FPR)
"""

import argparse, pandas as pd, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ───────────────────────── CLI ───────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--pred",  default="chemprop_canonical.csv")
ap.add_argument("--truth", default="experimental_labels.csv")
ap.add_argument("--id",    default="cid")
ap.add_argument("--step",  type=float, default=0.05)
ap.add_argument("--metric", choices=["f1","youden"], default="f1",
                help="criterion to pick best threshold")
args = ap.parse_args()

df = pd.merge(pd.read_csv(args.truth), pd.read_csv(args.pred), on=args.id)

endpoints = [c[len("True_"):] for c in df if c.startswith("True_")]
print(f"Endpoints found: {endpoints}")

for ep in endpoints:
    y_true = df[f"True_{ep}"].dropna()
    y_prob = df.loc[y_true.index, f"Pred_{ep}"]

    rows = []
    for th in np.arange(0, 1+args.step/2, args.step):
        y_hat = (y_prob >= th).astype(int)
        prec  = precision_score(y_true, y_hat, zero_division=0)
        rec   = recall_score(y_true, y_hat, zero_division=0)
        f1    = f1_score(y_true, y_hat, zero_division=0)
        youden= rec - (1-prec if prec>0 else 1)  # TPR-FPR proxy
        rows.append(dict(th=th, prec=prec, rec=rec, F1=f1, youden=youden))

    table = pd.DataFrame(rows)
    key   = "F1" if args.metric=="f1" else "youden"
    best  = table.loc[table[key].idxmax()]

    print(f"\n── {ep} ──  best {args.metric.upper()} at threshold {best.th:0.2f}")
    print(table.to_string(index=False, formatters={
        "th":    "{:0.2f}".format,
        "prec":  "{:0.3f}".format,
        "rec":   "{:0.3f}".format,
        "F1":    "{:0.3f}".format,
        "youden":"{:0.3f}".format
    }))
