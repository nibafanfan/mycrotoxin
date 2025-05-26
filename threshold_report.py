#!/usr/bin/env python3
# threshold_report.py
# -----------------------------------------------------------
# One-stop script to compare multiple probability thresholds
# for every endpoint and produce a clean report.
#
# • Reads chemprop_canonical.csv  +  experimental_labels.csv
# • Sweeps a grid (default 0.00 … 1.00, step 0.05)
# • Calculates Precision, Recall, F1, AUROC, AUPRC, BAC (=½·[TPR+TNR])
# • Writes results to threshold_report.csv
# • Prints a nice pivot table so you can copy-paste into a manuscript
#
# Usage:
#   python threshold_report.py            # default grid 0.05
#   python threshold_report.py --step 0.02 --out thresh_0p02.csv
# -----------------------------------------------------------

import argparse, numpy as np, pandas as pd, textwrap
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

EP = ["Carcinogenicity", "Mutagenicity", "Genotoxicity"]

def scan_thresholds(df, ep, grid):
    y = df[f"True_{ep}"].values.astype(int)
    p = df[f"Pred_{ep}"].values.astype(float)
    rows = []
    for t in grid:
        yhat = p >= t
        tp = ((y==1)&(yhat==1)).sum()
        fp = ((y==0)&(yhat==1)).sum()
        fn = ((y==1)&(yhat==0)).sum()
        tn = ((y==0)&(yhat==0)).sum()
        prec   = precision_score(y, yhat, zero_division=0)
        rec    = recall_score(y, yhat, zero_division=0)
        bac    = 0.5*(rec + (tn/(tn+fp) if (tn+fp) else 0))
        rows.append(dict(
            Endpoint = ep,
            Thres    = t,
            Prec     = prec,
            Recall   = rec,
            F1       = f1_score(y, yhat, zero_division=0),
            BAC      = bac,
            AUROC    = roc_auc_score(y, p) if len(np.unique(y))>1 else np.nan,
            AUPRC    = average_precision_score(y, p)
        ))
    return rows

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Generates a CSV with metrics for a grid of thresholds.
            Columns: Endpoint, Thres, Prec, Recall, F1, BAC, AUROC, AUPRC
        """))
    ap.add_argument("--pred",  default="chemprop_canonical.csv")
    ap.add_argument("--truth", default="experimental_labels.csv")
    ap.add_argument("--id",    default="cid")
    ap.add_argument("--step",  type=float, default=0.05,
                     help="grid step size (default 0.05)")
    ap.add_argument("--out",   default="threshold_report.csv",
                     help="output CSV file (default threshold_report.csv)")
    args = ap.parse_args()

    df = pd.merge(pd.read_csv(args.truth), pd.read_csv(args.pred), on=args.id)
    grid = np.arange(0.0, 1.00001, args.step)

    rows = []
    for ep in EP:
        mask = df[f"True_{ep}"].notna() & df[f"Pred_{ep}"].notna()
        rows.extend(scan_thresholds(df.loc[mask], ep, grid))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"✓ wrote {args.out}")

    # pretty pivot to screen
    show = (out.pivot(index="Thres", columns="Endpoint", values="F1")
                 .applymap(lambda x: f"{x:0.3f}"))
    print("\nF1 score at each threshold:")
    print(show.to_string())

if __name__ == "__main__":
    main()
