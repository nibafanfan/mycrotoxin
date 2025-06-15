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
- 1 if experimental data is exactly "Genotoxic"/"Carcinogenic"/"Mutagenic"
- 0 if experimental data is exactly "Non-genotoxic"/"Non-carcinogenic"/"Non-mutagenic"
- NaN for missing or other values

-----------------------------------------------------------------------
Usage
    python make_labels.py \
        --meta  mycotoxin_meta.tsv \
        --out   labels.csv
-----------------------------------------------------------------------
"""
import argparse
import pandas as pd
import pathlib
import sys
from typing import Optional

def label_genotoxicity(val: str) -> Optional[float]:
    """Label genotoxicity based on exact matches."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    if val == "Genotoxic":
        return 1.0
    if val == "Non-genotoxic":
        return 0.0
    return None

def label_carcinogenicity(val: str) -> Optional[float]:
    """Label carcinogenicity based on exact matches."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    if val == "Carcinogenic":
        return 1.0
    if val == "Non-carcinogenic":
        return 0.0
    return None

def label_mutagenicity(val: str) -> Optional[float]:
    """Label mutagenicity based on exact matches."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    if val == "Mutagenic":
        return 1.0
    if val == "Non-mutagenic":
        return 0.0
    return None

def main(meta_path: str, out_path: str):
    # Read and validate input file
    try:
        df = pd.read_csv(meta_path, sep=";")
    except Exception as e:
        sys.exit(f"Error reading metadata file: {e}")
    
    if "cid" not in df.columns:
        sys.exit("The metadata file must contain a 'cid' column.")
    
    required_cols = [
        "experimental in vitro genotoxicity",
        "experimental carcinogenicity",
        "experimental mutagenicity"
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        sys.exit(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Add label columns using exact matches
    df["genotoxicity"] = df["experimental in vitro genotoxicity"].apply(label_genotoxicity)
    df["carcinogenicity"] = df["experimental carcinogenicity"].apply(label_carcinogenicity)
    df["mutagenicity"] = df["experimental mutagenicity"].apply(label_mutagenicity)
    
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
    
    # Save output
    try:
        df[out_cols].to_csv(out_path, index=False)
    except Exception as e:
        sys.exit(f"Error saving output file: {e}")
    
    # Print summary statistics
    print(f"\n✔ Labels saved to: {pathlib.Path(out_path).resolve()}")
    print(f"  Total rows: {df.shape[0]}")
    
    for prop in ["genotoxicity", "carcinogenicity", "mutagenicity"]:
        active = df[prop].sum()
        inactive = (df[prop] == 0).sum()
        missing = df[prop].isna().sum()
        print(f"\n{prop.title()}:")
        print(f"  • Active (1): {active:.0f}")
        print(f"  • Inactive (0): {inactive:.0f}")
        print(f"  • Missing: {missing}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert experimental data to binary labels")
    ap.add_argument("--meta", required=True, help="semicolon-separated metadata table")
    ap.add_argument("--out", default="labels.csv", help="output labels file")
    args = ap.parse_args()
    main(args.meta, args.out)
