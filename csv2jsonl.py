#!/usr/bin/env python3
# csv2jsonl.py – wide CSV ➜ long JSONL (no API calls)

import pandas as pd, json, re, unicodedata

WIDE = "chemprop_preds.csv"
OUT  = "chemprop_preds.jsonl"

def unslug(label):
    """SMILES_potency_score → 'SMILES potency score'"""
    s = unicodedata.normalize("NFKD", str(label))
    return re.sub(r"_+", " ", s).strip()

df = pd.read_csv(WIDE)
pred_cols = [c for c in df.columns if c not in ("cid", "inchi")]

with open(OUT, "w") as jf:
    for _, row in df.iterrows():
        preds = []
        for col in pred_cols:
            val = row[col]
            if pd.notna(val):
                preds.append({
                    "value": float(val),
                    "property": {
                        "categories": [{"category": unslug(col)}]
                    }
                })
        jf.write(json.dumps({"cid": int(row["cid"]),
                             "inchi": row["inchi"],
                             "pred": preds}) + "\n")

print("✓ wrote", len(df), "records to", OUT)
