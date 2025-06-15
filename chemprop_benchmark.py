#!/usr/bin/env python3
# chemprop_benchmark.py
"""
Compare two feature-selection strategies for multiple endpoints
(genotoxicity, carcinogenicity, mutagenicity).

Strategy A  – only those property_tokens whose category list contains
              any of {carcinogenicity, mutagenicity, genotoxicity}
Strategy B  – every prediction column in the file

Both are evaluated with the same elastic-net logistic model under
stratified k-fold cross-validation (default k = 5).

---------------------------------------------------------------------------
Inputs
  • chemprop_flat.parquet   – produced by chemprop_pipeline.py
  • labels.csv              – columns: index, genotoxicity, carcinogenicity, mutagenicity

Example labels.csv
    index,genotoxicity,carcinogenicity,mutagenicity
    1677,1,0,1
    1021,0,1,0
    …

Run
    python chemprop_benchmark.py \
        --parquet chemprop_flat.parquet \
        --labels  labels.csv \
        --kfolds  5 \
        --seed    42
---------------------------------------------------------------------------

Requires: pandas, numpy, scikit-learn, tqdm, pyarrow
"""
import argparse, json, collections, pandas as pd, numpy as np, pathlib, warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)
CATS = {"carcinogenicity", "mutagenicity", "genotoxicity"}
ENDPOINTS = ["genotoxicity", "carcinogenicity", "mutagenicity"]

# -------------------------------------------------------------------------- #
# 1. Build token → categories lookup once from the JSON baked into parquet
# -------------------------------------------------------------------------- #
def token_catalog(df: pd.DataFrame) -> pd.DataFrame:
    # parquet has no category info; pull it out of a single embedded json cell
    # find the first non-null record that still holds raw API json (kept by pipeline)
    raw_json = None
    for col in ("api_response_json", "api_response"):   # back-compat
        if col in df.columns:
            raw_json = df[col].dropna().iloc[0]
            break
    if raw_json is None:
        raise RuntimeError("The parquet no longer contains raw API JSON; "
                           "cannot reconstruct category list.")
    import json
    rec = json.loads(raw_json)    # one molecule is enough
    # rec structure: api_response is list; each has property_token & categories
    tok2cats = {p["property_token"]:
                ";".join(sorted({c["category"] for c in p["property"]["categories"]}))
                for p in rec["api_response"]}
    return pd.DataFrame({"tok": list(tok2cats),
                         "cats": list(tok2cats.values())})

# -------------------------------------------------------------------------- #
# 2. Model evaluation
# -------------------------------------------------------------------------- #
def cv_eval(X: pd.DataFrame, y: pd.Series, tag: str, k: int, seed: int):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs, baccs = [], []
    for tr, te in skf.split(X, y):
        y_tr = y.iloc[tr]
        if len(y_tr.unique()) < 2:
            print(f"Warning: Fold contains only one class. Skipping fold.")
            continue
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(solver="saga", penalty="elasticnet",
                               l1_ratio=0.5, max_iter=4000,
                               random_state=seed),
        )
        pipe.fit(X.iloc[tr], y_tr)
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        preds = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y.iloc[te], proba))
        baccs.append(balanced_accuracy_score(y.iloc[te], preds))
    if aucs:
        print(f"{tag:<6}| AUROC {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
              f" | BACC {np.mean(baccs):.3f} | n_feat={X.shape[1]}")
    else:
        print(f"{tag:<6}| No valid folds found.")

# -------------------------------------------------------------------------- #
def main(parquet: str, labels: str, kfolds: int, seed: int):
    df = pd.read_parquet(parquet)
    labels_df = pd.read_csv(labels)
    
    # Merge on 'cid'
    if 'cid' not in df.columns or 'cid' not in labels_df.columns:
        raise ValueError("Both feature and label files must have a 'cid' column.")
    merged = pd.merge(df, labels_df, on='cid', how='inner')
    
    tok_tbl = token_catalog(df)
    subset_tok = tok_tbl.loc[
        tok_tbl.cats.str.contains("|".join(CATS), case=False, na=False),
        "tok"
    ]

    subset_cols = [f"pred_{t}" for t in subset_tok if f"pred_{t}" in df.columns]
    all_cols    = [c for c in df.columns if c.startswith("pred_")]

    # drop columns with <10 % non-null values
    min_n = 0.1 * len(df)
    X_sub = merged[subset_cols].loc[:, merged[subset_cols].notna().sum() >= min_n]
    X_all = merged[all_cols].loc[:,   merged[all_cols].notna().sum()   >= min_n]

    print(f"{len(merged)} molecules  |  subset features {X_sub.shape[1]}  |  all features {X_all.shape[1]}")
    
    # Handle each endpoint separately
    for endpoint in ENDPOINTS:
        print(f"\n{endpoint.upper()}:")
        y = merged[endpoint].astype(float)
        mask = y.notna()
        if mask.sum() < 2:
            print(f"Not enough samples for {endpoint}. Skipping.")
            continue
        cv_eval(X_sub[mask], y[mask], "SUBSET", kfolds, seed)
        cv_eval(X_all[mask], y[mask], "ALL",    kfolds, seed)

# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="chemprop_flat.parquet")
    ap.add_argument("--labels",  required=True, help="labels.csv (index,genotoxicity,carcinogenicity,mutagenicity)")
    ap.add_argument("--kfolds",  type=int, default=5)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()
    main(args.parquet, args.labels, args.kfolds, args.seed)
