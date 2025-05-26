#!/usr/bin/env python3
"""
eval_tox_pred.py  –  evaluate ChemProp probabilities vs experimental truth
Adds per-endpoint thresholds via  --thres list  OR  --yaml thresholds.yml
"""

import argparse, pandas as pd, numpy as np, yaml, sys
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

EP = ["Carcinogenicity", "Mutagenicity", "Genotoxicity"]

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

def metrics(y, p, thr):
    y_hat = (p >= thr).astype(int)
    return dict(
        AUROC  = safe_auc(y, p),
        AUPRC  = average_precision_score(y, p),
        Acc    = (y == y_hat).mean(),
        Prec   = precision_score(y, y_hat, zero_division=0),
        Recall = recall_score(y, y_hat, zero_division=0),
        F1     = f1_score(y, y_hat, zero_division=0),
        TP     = int(((y==1)&(y_hat==1)).sum()),
        FP     = int(((y==0)&(y_hat==1)).sum()),
        FN     = int(((y==1)&(y_hat==0)).sum()),
        TN     = int(((y==0)&(y_hat==0)).sum()),
    )

def parse_thr_arg(arg):
    vals = [float(x) for x in arg.split(",")]
    if len(vals) == 1:                  # single value → broadcast
        vals *= len(EP)
    if len(vals) != len(EP):
        sys.exit(f"--thres needs 1 or {len(EP)} values (got {vals})")
    return dict(zip(EP, vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",  required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--id",    default="cid")
    ap.add_argument("--thres", default="0.5",
                    help="single value or comma list (Carc,Muta,Geno)")
    ap.add_argument("--yaml",  help="YAML with thresholds dict")
    a = ap.parse_args()

    thr = parse_thr_arg(a.thres)
    if a.yaml:
        yml = yaml.safe_load(open(a.yaml))
        if "thresholds" in yml:
            thr.update({k.capitalize():v for k,v in yml["thresholds"].items()})

    df = pd.merge(pd.read_csv(a.truth), pd.read_csv(a.pred), on=a.id)

    rows = []
    for ep in EP:
        y_col = f"True_{ep}"
        p_col = f"Pred_{ep}"
        mask  = df[y_col].notna() & df[p_col].notna()
        y     = df.loc[mask, y_col].astype(int).values
        p     = df.loc[mask, p_col].values
        m     = metrics(y, p, thr[ep])
        rows.append([ep, len(y), *[f"{m[k]:.3f}" if isinstance(m[k], float) else m[k]
                                   for k in ("AUROC","AUPRC","Acc","Prec","Recall","F1",
                                             "TP","FP","FN","TN")]])
    out = pd.DataFrame(rows, columns=["Endpoint","N","AUROC","AUPRC","Acc",
                                      "Prec","Recall","F1","TP","FP","FN","TN"])
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
