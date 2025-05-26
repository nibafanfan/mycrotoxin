#!/usr/bin/env python3
"""
eval_tox_pred.py – compare ChemProp predictions with experimental truth.
• Reads any number of *True_* columns from the truth file.
• Expects matching *Pred_* columns in the prediction file
  (same suffix after the underscore, case-insensitive).
• Drops rows with missing truth for that endpoint.
• Reports AUROC, AUPRC, accuracy, precision, recall, F1 and the
  confusion-matrix counts for each endpoint.
"""

import argparse, pandas as pd, numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from tabulate import tabulate

def binarise(arr, th): return (arr >= th).astype(int)

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

def safe_auprc(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan

def metrics(y_true, y_prob, th):
    y_hat = binarise(y_prob, th)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return dict(
        N      = len(y_true),
        AUROC  = safe_auc(y_true, y_prob),
        AUPRC  = safe_auprc(y_true, y_prob),
        Acc    = accuracy_score(y_true, y_hat),
        Prec   = precision_score(y_true, y_hat, zero_division=0),
        Recall = recall_score(y_true, y_hat, zero_division=0),
        F1     = f1_score(y_true, y_hat, zero_division=0),
        TP=tp, FP=fp, FN=fn, TN=tn,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",  required=True, help="prediction CSV")
    ap.add_argument("--truth", required=True, help="ground-truth CSV")
    ap.add_argument("--id",    default="cid", help="column to merge on")
    ap.add_argument("--thres", type=float, default=0.5,
                    help="probability threshold (default 0.5)")
    ap.add_argument("--out",   default="eval_report.tsv",
                    help="where to write the TSV report")
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    pred  = pd.read_csv(args.pred)

    df = pd.merge(truth, pred, on=args.id, how="inner")
    if df.empty:
        raise SystemExit("❌  No overlapping IDs between truth and prediction.")

    # discover all endpoints automatically
    endpoints = [c[len("True_"):] for c in df.columns if c.startswith("True_")]
    if not endpoints:
        raise SystemExit("❌  No *True_* columns found in truth table.")

    rows = []
    for ep in endpoints:
        y_col = f"True_{ep}"
        p_col = [c for c in df.columns if c.lower() == f"pred_{ep.lower()}"]
        if not p_col:
            print(f"⚠️  skipping {ep}: no matching Pred_{ep} column")
            continue
        p_col = p_col[0]

        mask = df[y_col].notna() & df[p_col].notna()
        y_true = df.loc[mask, y_col].astype(int).values
        y_prob = df.loc[mask, p_col].astype(float).values

        if len(y_true) == 0:
            print(f"⚠️  skipping {ep}: all labels are NA")
            continue

        m = metrics(y_true, y_prob, args.thres)
        m["Endpoint"] = ep
        rows.append(m)

    if not rows:
        raise SystemExit("❌  No endpoints evaluated (see warnings above).")

    cols = ["Endpoint","N","AUROC","AUPRC","Acc",
            "Prec","Recall","F1","TP","FP","FN","TN"]
    print(tabulate([[r.get(c,np.nan) for c in cols] for r in rows],
                   headers=cols, floatfmt=".3f"))
    pd.DataFrame(rows)[cols].to_csv(args.out, sep="\t", index=False)
    print(f"\n✓ report saved → {args.out}")

if __name__ == "__main__":
    main()
