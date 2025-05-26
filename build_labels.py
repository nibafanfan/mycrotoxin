#!/usr/bin/env python3
"""
build_labels.py – extract experimental ground-truth labels
from mycotoxins_tmap_final.csv and save experimental_labels.csv
ready for eval_tox_pred.py.
"""

import pandas as pd

IN  = "mycotoxins_tmap_final.csv"   # semicolon-delimited source file
OUT = "experimental_labels.csv"     # truth table the evaluator will read

# map textual calls → binary 1 / 0
MAP = {
    "Carcinogenic"     : 1,
    "Non-carcinogenic" : 0,
    "Mutagenic"        : 1,
    "Non-mutagenic"    : 0,
    "Genotoxic"        : 1,
    "Non-genotoxic"    : 0,
}

df = pd.read_csv(IN, sep=";")

labels = pd.DataFrame({
    "cid": df["cid"],
    "True_Carcinogenicity":
        df["experimental carcinogenicity"].map(MAP).astype("Int64"),
    "True_Mutagenicity":
        df["experimental mutagenicity"].map(MAP).astype("Int64"),
    "True_Genotoxicity":        # use the in-vitro column as truth
        df["experimental in vitro genotoxicity"].map(MAP).astype("Int64"),
})

labels.to_csv(OUT, index=False)
print(f"✓ truth table written → {OUT}")
print(labels.head())
