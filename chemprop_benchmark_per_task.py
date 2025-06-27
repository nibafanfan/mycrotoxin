#!/usr/bin/env python3
"""
chemprop_benchmark_per_task.py  –  per-endpoint benchmark on ChemProp /
ToxTransformer prediction columns.

For each endpoint (genotoxicity, carcinogenicity, mutagenicity) it:

1. Splits the data once into       TRAIN (80 %) / TEST (20 %)  stratified.
2. On TRAIN:
   • selects the top-k features by mutual information with the label
   • fits a StandardScaler + LogisticRegressionCV (elastic-net, l1_ratio = 0.5)
   • reports cross-validated AUROC ± σ and BACC on the TRAIN portion.
3. Evaluates the fitted model on TEST (hold-out) and reports AUROC, BACC.
4. Prints the top-N features
   • by mutual-information score (univariate)
   • by absolute coefficient weight in the trained elastic-net (multivariate)

Run
    python chemprop_benchmark_per_task.py \
        --parquet chemprop_flat.parquet \
        --labels  labels.csv \
        --topk    500 \
        --seed    42
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ENDPOINTS = ["genotoxicity", "carcinogenicity", "mutagenicity"]
C_GRID    = np.logspace(-3, 1, 8)             # 0.001 … 10
MAX_ITER  = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# util helpers
# ─────────────────────────────────────────────────────────────────────────────
def token_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Extract token→property mapping from first api_response cell."""
    col = next(c for c in ("api_response_json", "api_response") if c in df)
    rec = json.loads(df[col].dropna().iloc[0])
    return {
        p["property_token"]: p["property"]["name"]
        for p in rec["api_response"]
    }


def evaluate_cv(model, X, y, kfolds, repeats, seed):
    """Return mean±std AUROC and mean BACC via repeated CV on TRAIN only."""
    rkf = RepeatedStratifiedKFold(
        n_splits=kfolds, n_repeats=repeats, random_state=seed
    )
    aucs, baccs = [], []
    for tr, te in rkf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        preds = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y.iloc[te], proba))
        baccs.append(balanced_accuracy_score(y.iloc[te], preds))
    return np.mean(aucs), np.std(aucs), np.mean(baccs)


def get_top_features(X_train, selector, model, topn=20):
    """Return (mi_df, coef_df) each with columns [feature, score/coef]."""
    # MI
    mi_scores = selector.scores_[selector.get_support()]
    names     = X_train.columns[selector.get_support()]
    mi_df = (
        pd.DataFrame({"feature": names, "mutual_info": mi_scores})
        .sort_values("mutual_info", ascending=False)
        .head(topn)
        .reset_index(drop=True)
    )

    # Coefficient magnitude
    coefs = model.named_steps["logisticregressioncv"].coef_.flatten()
    coef_df = (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_weight=lambda d: d.coefficient.abs())
        .sort_values("abs_weight", ascending=False)
        .head(topn)
        .reset_index(drop=True)
    )
    return mi_df, coef_df


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main(
    parquet: str,
    labels: str,
    topk: int,
    seed: int,
    test_size: float,
    kfolds: int,
    repeats: int,
    topn: int,
):
    feats_df  = pd.read_parquet(parquet)
    labels_df = pd.read_csv(labels)

    merged = feats_df.merge(labels_df, on="cid", how="inner")
    feat_cols = [c for c in merged.columns if c.startswith("pred_")]

    # drop columns with too many nulls (>=10 % present)
    min_n = 0.1 * len(merged)
    avail = merged[feat_cols].loc[:, merged[feat_cols].notna().sum() >= min_n]

    print(f"{len(merged)} molecules | {avail.shape[1]} raw features\n")

    for ep in ENDPOINTS:
        print(f"{'='*60}\n{ep.upper()}")
        y_all = merged[ep].astype(float)
        mask  = y_all.notna()
        y     = y_all[mask]
        X     = avail.loc[mask]

        if y.nunique() < 2:
            print("! Skipped (not enough class balance)\n"); continue

        # split hold-out
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )

        # feature selection fit only on TRAIN
        selector = SelectKBest(mutual_info_classif, k=min(topk, X_tr.shape[1]))
        X_tr_sel = selector.fit_transform(X_tr, y_tr)
        kept     = X_tr.columns[selector.get_support()]

        # transform TEST
        X_te_sel = selector.transform(X_te)

        # wrap back into DataFrames for convenience
        X_tr_sel = pd.DataFrame(X_tr_sel, index=y_tr.index, columns=kept)
        X_te_sel = pd.DataFrame(X_te_sel, index=y_te.index, columns=kept)

        # model pipeline
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegressionCV(
                Cs=C_GRID,
                penalty="elasticnet",
                solver="saga",
                l1_ratios=[0.5],
                class_weight="balanced",
                scoring="roc_auc",
                max_iter=MAX_ITER,
                n_jobs=-1,
                random_state=seed,
            ),
        )

        # CV on TRAIN
        auc_m, auc_s, bacc_m = evaluate_cv(
            pipe, X_tr_sel, y_tr, kfolds, repeats, seed
        )
        print(f"CV   | AUROC {auc_m:.3f}±{auc_s:.3f} | BACC {bacc_m:.3f}")

        # fit on full TRAIN & eval on TEST
        pipe.fit(X_tr_sel, y_tr)
        proba_te = pipe.predict_proba(X_te_sel)[:, 1]
        preds_te = (proba_te >= 0.5).astype(int)
        auc_te   = roc_auc_score(y_te, proba_te)
        bacc_te  = balanced_accuracy_score(y_te, preds_te)
        print(f"TEST | AUROC {auc_te:.3f}          | BACC {bacc_te:.3f}")

        # feature importance tables
        mi_df, coef_df = get_top_features(X_tr, selector, pipe, topn=topn)

        print("\nTop features by Mutual Information")
        print(mi_df.to_string(index=False))
        print("\nTop features by |Coefficient|")
        print(coef_df.to_string(index=False))
        print()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--labels",  required=True)
    ap.add_argument("--topk",    type=int, default=500,
                    help="num features per endpoint after MI selection")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="hold-out fraction (0.2 = 20 %)")
    ap.add_argument("--kfolds",  type=int, default=5,
                    help="folds for CV within TRAIN")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats for RepeatedStratifiedKFold")
    ap.add_argument("--topn",    type=int, default=20,
                    help="how many top features to print")
    args = ap.parse_args()

    main(
        args.parquet,
        args.labels,
        args.topk,
        args.seed,
        args.test_size,
        args.kfolds,
        args.repeats,
        args.topn,
    )
