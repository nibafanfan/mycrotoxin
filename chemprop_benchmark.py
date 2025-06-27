#!/usr/bin/env python3
"""
chemprop_benchmark.py  –  compare two ChemProp feature sets under elastic‑net
logistic regression with nested cross‑validation.

Strategy A  =  only tokens whose category list mentions {carcinogenicity,
                 mutagenicity, genotoxicity}
Strategy B  =  all ~4 000 ChemProp prediction columns

Changes (2025‑06‑15)
────────────────────
* Replaced plain `LogisticRegression` with **`LogisticRegressionCV`** to let
  scikit‑learn find the optimal C value per fold and to run until convergence
  (option 2 in our earlier discussion).  This eliminates `ConvergenceWarning`
  spam without loosening tolerances or increasing max_iter globally.
* Added a common grid `Cs = [0.02, 0.1, 0.5, 1]` that mirrors the selector
  grid in the main pipeline.

Run example
    python chemprop_benchmark.py \
        --parquet chemprop_flat.parquet \
        --labels  labels.csv \
        --kfolds  5 \
        --seed    42
"""
import argparse, json, pathlib, warnings
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

CATS       = {"carcinogenicity", "mutagenicity", "genotoxicity"}
ENDPOINTS  = ["genotoxicity", "carcinogenicity", "mutagenicity"]
C_GRID     = [0.002, 0.01, 0.05, 0.1]       # shared across all folds
MAX_ITER   = 10_000

# ──────────────────────────────────────────────────────────────────────────────
# helper: build token→category table from embedded JSON
# ──────────────────────────────────────────────────────────────────────────────

def token_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns [tok, cats] extracted from first JSON cell."""
    col = next((c for c in ("api_response_json", "api_response") if c in df), None)
    if col is None:
        raise RuntimeError("No raw API JSON found; cannot recover categories.")
    rec = json.loads(df[col].dropna().iloc[0])  # one compound is enough
    pairs = {
        p["property_token"]: ";".join(sorted({c["category"] for c in p["property"]["categories"]}))
        for p in rec["api_response"]
    }
    return pd.DataFrame({"tok": list(pairs), "cats": list(pairs.values())})

# ──────────────────────────────────────────────────────────────────────────────
# modelling util – uses LogisticRegressionCV (elastic‑net with L1 ratio 0.5)
# ──────────────────────────────────────────────────────────────────────────────

def cv_eval(X: pd.DataFrame, y: pd.Series, tag: str, k: int, seed: int):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs, baccs = [], []
    for tr, te in skf.split(X, y):
        y_tr = y.iloc[tr]
        if y_tr.nunique() < 2:
            continue  # skip degenerate folds
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegressionCV(
                Cs=C_GRID,
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                l1_ratios=[0.5],
                max_iter=MAX_ITER,
                n_jobs=-1,
                random_state=seed,
                scoring="roc_auc",
            ),
        )
        pipe.fit(X.iloc[tr], y_tr)
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        preds = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y.iloc[te], proba))
        baccs.append(balanced_accuracy_score(y.iloc[te], preds))
    if aucs:
        print(f"{tag:<6}| AUROC {np.mean(aucs):.3f}±{np.std(aucs):.3f} | "
              f"BACC {np.mean(baccs):.3f} | n_feat={X.shape[1]}")
    else:
        print(f"{tag:<6}| No valid folds")

# ──────────────────────────────────────────────────────────────────────────────
# main routine
# ──────────────────────────────────────────────────────────────────────────────

def main(parquet: str, labels: str, kfolds: int, seed: int):
    df        = pd.read_parquet(parquet)
    labels_df = pd.read_csv(labels)

    if "cid" not in df.columns or "cid" not in labels_df.columns:
        raise ValueError("Both files must have a 'cid' column for merging.")
    merged = df.merge(labels_df, on="cid", how="inner")

    tok_tbl = token_catalog(df)
    subset_tok = tok_tbl.loc[
        tok_tbl.cats.str.contains("|".join(CATS), case=False, na=False), "tok"
    ]

    subset_cols = [f"pred_{t}" for t in subset_tok if f"pred_{t}" in merged.columns]
    all_cols    = [c for c in merged.columns if c.startswith("pred_")]

    min_n = 0.1 * len(merged)  # keep cols with ≥10 % non‑null
    X_sub = merged[subset_cols].loc[:, merged[subset_cols].notna().sum() >= min_n]
    X_all = merged[all_cols].loc[:,   merged[all_cols].notna().sum()   >= min_n]

    print(f"{len(merged)} molecules | subset {X_sub.shape[1]} feats | all {X_all.shape[1]} feats")

    for endpoint in ENDPOINTS:
        print(f"\n{endpoint.upper()}:")
        y = merged[endpoint].astype(float)
        mask = y.notna()
        if mask.sum() < 2:
            print("Not enough samples – skipped"); continue
        cv_eval(X_sub[mask], y[mask], "SUBSET", kfolds, seed)
        cv_eval(X_all[mask], y[mask], "ALL",    kfolds, seed)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--labels",  required=True)
    ap.add_argument("--kfolds",  type=int, default=5)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()
    main(args.parquet, args.labels, args.kfolds, args.seed)
