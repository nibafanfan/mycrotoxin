# Minimal version of merge_pred.py with clean column handling
import pandas as pd
import re

RAW = "mycotoxins_tmap_final.csv"
PRED = "success.csv"
OUT = "mycotoxin_chemprop_comparison.csv"

def canon_name(txt):
    if pd.isna(txt): return ""
    txt = str(txt).lower()
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("–", "-")
    return txt.strip()

# Load and normalize
raw = pd.read_csv(RAW, sep=";")  # use proper single-character sep
pred = pd.read_csv(PRED)

raw["__key__"] = raw["name"].apply(canon_name)
pred["__key__"] = pred["name"].apply(canon_name)

# Keep only prediction-relevant columns from pred
keep_cols = ["__key__", "InChI", "Pred_Mutagenicity", "Pred_Genotoxicity", "Pred_Carcinogenicity"]
pred = pred[keep_cols]

# Merge cleanly
merged = raw.merge(pred, on="__key__", how="left").drop(columns="__key__")
merged.to_csv(OUT, index=False)

print(f"✅ merged → {OUT}   (rows: {len(merged)})")
