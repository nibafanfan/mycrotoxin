# make_labels.py
"""
Convert the author-supplied semicolon table into labels.csv, keeping only rows
that have experimental values (not missing) and outputting only:
- cid
- experimental in vitro genotoxicity
- experimental carcinogenicity
- experimental mutagenicity
- genotoxicity
- carcinogenicity
- mutagenicity

Each label is binary (0/1):
- 1 if experimental data contains "genotoxic"/"carcinogenic"/"mutagenic"
- 0 if experimental data contains "non-genotoxic"/"non-carcinogenic"/"non-mutagenic"

-----------------------------------------------------------------------
Usage
    python make_labels.py \
        --meta  mycotoxin_meta.tsv \
        --out   labels.csv
-----------------------------------------------------------------------
"""
import argparse, pandas as pd, re, pathlib, sys

def label_genotoxicity(row) -> float:
    val = str(row["experimental in vitro genotoxicity"]).lower()
    if not val or val == "nan" or pd.isna(val):
        return float("nan")
    if "genotoxic" in val:
        return 1.0
    if "non-genotoxic" in val:
        return 0.0
    return float("nan")

def label_carcinogenicity(row) -> float:
    val = str(row["experimental carcinogenicity"]).lower()
    if not val or val == "nan" or pd.isna(val):
        return float("nan")
    if "carcinogenic" in val:
        return 1.0
    if "non-carcinogenic" in val:
        return 0.0
    return float("nan")

def label_mutagenicity(row) -> float:
    val = str(row["experimental mutagenicity"]).lower()
    if not val or val == "nan" or pd.isna(val):
        return float("nan")
    if "mutagenic" in val:
        return 1.0
    if "non-mutagenic" in val:
        return 0.0
    return float("nan")

def main(meta_path: str, out_path: str):
    df = pd.read_csv(meta_path, sep=";")
    if "cid" not in df.columns:
        sys.exit("The metadata file must contain a 'cid' column.")
    
    # Add label columns
    df["genotoxicity"] = df.apply(label_genotoxicity, axis=1)
    df["carcinogenicity"] = df.apply(label_carcinogenicity, axis=1)
    df["mutagenicity"] = df.apply(label_mutagenicity, axis=1)
    
    # Keep only rows that have at least one experimental value
    has_experimental = (
        df["experimental in vitro genotoxicity"].notna() |
        df["experimental carcinogenicity"].notna() |
        df["experimental mutagenicity"].notna()
    )
    df = df[has_experimental]
    
    # Output only the requested columns
    out_cols = [
        "cid",
        "experimental in vitro genotoxicity",
        "experimental carcinogenicity",
        "experimental mutagenicity",
        "genotoxicity",
        "carcinogenicity",
        "mutagenicity"
    ]
    df[out_cols].to_csv(out_path, index=False)
    
    print(f"✔ labels → {pathlib.Path(out_path).resolve()}  ({df.shape[0]} rows)")
    print(f"  • genotoxicity: {df['genotoxicity'].sum():.0f} active, {df['genotoxicity'].isna().sum()} missing")
    print(f"  • carcinogenicity: {df['carcinogenicity'].sum():.0f} active, {df['carcinogenicity'].isna().sum()} missing")
    print(f"  • mutagenicity: {df['mutagenicity'].sum():.0f} active, {df['mutagenicity'].isna().sum()} missing")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="semicolon-separated metadata table")
    ap.add_argument("--out",  default="labels.csv", help="output labels file")
    args = ap.parse_args()
    main(args.meta, args.out)
