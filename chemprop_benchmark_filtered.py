#!/usr/bin/env python3
# chemprop_benchmark_filtered.py
#
# ▸ removes “leaky” columns   (property name / category contains endpoint)
# ▸ 80 / 20 hold-out split   (stratified)
# ▸ Repeated 5-fold CV on TRAIN
# ▸ Top-k (500) MI features per endpoint
# ▸ Elastic-net LogisticRegressionCV
# ▸ Bar-plot of |coef|  (top 20)
# ▸ CSV with decoded feature ↔ property metadata           ───────────────

import argparse, json, pathlib, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def token_catalog(df: pd.DataFrame) -> dict:
    col = next(c for c in ("api_response_json", "api_response") if c in df)
    rec = json.loads(df[col].dropna().iloc[0])
    d = {}
    for p in rec["api_response"]:
        tok = str(p["property_token"])
        d[tok] = {
            "name": p["property"]["name"],
            "cats": [c["category"] for c in p["property"]["categories"]],
        }
    return d


def is_leaky(tok: str, target: str, meta: dict) -> bool:
    m = meta.get(tok, {})
    t = target.lower()
    return (
        t in m.get("name", "").lower()
        or any(t in c.lower() for c in m.get("cats", []))
    )


def decode(feat: str, meta: dict):
    tok = feat.removeprefix("pred_")
    m = meta.get(tok, {"name": "UNKNOWN", "cats": []})
    return m["name"], ";".join(m["cats"])


# ────────────────────────────────────────────────────────────────────────────
def run_one_endpoint(ep, X, y, meta, topk=500, seed=42):
    # ↓ 80 / 20 hold-out
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    sel = SelectKBest(mutual_info_classif, k=min(topk, X_tr.shape[1]))
    X_tr_sel = sel.fit_transform(X_tr, y_tr)
    kept_idx = sel.get_support(indices=True)
    kept_cols = X_tr.columns[kept_idx]
    X_te_sel = sel.transform(X_te)

    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 8),
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.5],
            class_weight="balanced",
            scoring="roc_auc",
            max_iter=10_000,
            n_jobs=-1,
            random_state=seed,
        ),
    )

    rkf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=seed
    )
    aucs, baccs = [], []
    for tr, te in rkf.split(X_tr_sel, y_tr):
        pipe.fit(X_tr_sel[tr], y_tr.iloc[tr])
        proba = pipe.predict_proba(X_tr_sel[te])[:, 1]
        preds = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_tr.iloc[te], proba))
        baccs.append(balanced_accuracy_score(y_tr.iloc[te], preds))

    pipe.fit(X_tr_sel, y_tr)
    proba_te = pipe.predict_proba(X_te_sel)[:, 1]
    preds_te = (proba_te >= 0.5).astype(int)

    # performance summary
    perf = dict(
        endpoint=ep,
        cv_auroc=np.mean(aucs),
        cv_std=np.std(aucs),
        cv_bacc=np.mean(baccs),
        test_auroc=roc_auc_score(y_te, proba_te),
        test_bacc=balanced_accuracy_score(y_te, preds_te),
        n_features=X_tr_sel.shape[1],
    )

    # top-20 coefficients
    clf = pipe.named_steps["logisticregressioncv"]
    coefs = clf.coef_.flatten()
    abs_coefs = np.abs(coefs)
    order = abs_coefs.argsort()[::-1][:20]

    top_rows = []
    mi_kept = sel.scores_[kept_idx]
    for rank, idx in enumerate(order, 1):
        feat = kept_cols[idx]
        name, cats = decode(feat, meta)
        top_rows.append(
            dict(
                endpoint=ep,
                rank=rank,
                feature=feat,
                property_name=name,
                categories=cats,
                coefficient=coefs[idx],
                abs_coefficient=abs_coefs[idx],
                mutual_info=mi_kept[idx],
            )
        )

    # bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(
        range(20),
        [r["abs_coefficient"] for r in top_rows],
    )
    plt.xticks(
        range(20),
        [r["feature"].replace("pred_", "") for r in top_rows],
        rotation=90,
    )
    plt.ylabel("abs(coefficient)")
    plt.title(f"Top 20 |coef| – {ep}")
    plt.tight_layout()
    plt.show()

    return perf, top_rows


# ────────────────────────────────────────────────────────────────────────────
def main(parquet, labels, topk=500):
    # Needs pyarrow or fastparquet
    try:
        df_feat = pd.read_parquet(parquet)
    except Exception as e:
        sys.exit(
            f"❌  Could not read {parquet} → install `pyarrow` "
            f"or `fastparquet` and retry.\n{e}"
        )

    df_lbl = pd.read_csv(labels)
    merged = df_feat.merge(df_lbl, on="cid", how="inner")

    feat_cols = [c for c in merged.columns if c.startswith("pred_")]
    min_n = 0.1 * len(merged)
    avail = merged[feat_cols].loc[
        :, merged[feat_cols].notna().sum() >= min_n
    ]

    meta = token_catalog(df_feat)

    all_perf, all_rows = [], []
    for ep in ["genotoxicity", "carcinogenicity", "mutagenicity"]:
        y = merged[ep].astype(float)
        mask = y.notna()
        X = avail.loc[mask]

        # drop leaky columns
        bad_cols = [
            f"pred_{tok}"
            for tok in meta
            if is_leaky(tok, ep, meta)
        ]
        X = X.drop(columns=bad_cols, errors="ignore")
        y = y[mask]

        if y.nunique() < 2:
            print(f"⚠  Skipping {ep} (class imbalance)")
            continue

        perf, rows = run_one_endpoint(ep, X, y, meta, topk=topk)
        all_perf.append(perf)
        all_rows.extend(rows)

    # summary
    print("\n───────── PERFORMANCE (filtered) ─────────")
    print(pd.DataFrame(all_perf).to_string(index=False))

    out_csv = pathlib.Path("top_features_decoded.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n✓  Feature table written → {out_csv.resolve()}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--topk", type=int, default=500)
    args = ap.parse_args()

    main(args.parquet, args.labels, topk=args.topk)
