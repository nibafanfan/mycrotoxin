#!/usr/bin/env python3
"""
deduplicate_chemprop.py - Remove duplicate compounds from chemprop_flat.parquet

This script removes duplicate compounds based on InChI strings and keeps only
the first occurrence of each unique compound.
"""
import argparse
import pathlib
import pandas as pd

def deduplicate_chemprop(input_path: str, output_path: str) -> None:
    """Remove duplicate compounds from the chemprop data."""
    
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original unique compounds: {df['InChI'].nunique()}")
    
    # Count duplicates
    duplicates = df.duplicated(subset=['InChI'], keep=False)
    duplicate_count = duplicates.sum()
    print(f"Duplicate compounds found: {duplicate_count}")
    
    if duplicate_count > 0:
        # Show some examples of duplicates
        print("\nExample duplicates:")
        duplicate_examples = df[duplicates].groupby('InChI').head(2)
        for inchi in duplicate_examples['InChI'].unique()[:3]:
            dup_rows = df[df['InChI'] == inchi]
            print(f"InChI: {inchi[:50]}...")
            print(f"  Names: {list(dup_rows['name'])}")
            print(f"  CIDs: {list(dup_rows['cid'])}")
            print()
    
    # Remove duplicates, keeping first occurrence
    df_clean = df.drop_duplicates(subset=['InChI'], keep='first')
    
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Compounds removed: {len(df) - len(df_clean)}")
    
    # Save cleaned data
    print(f"Saving cleaned data to {output_path}...")
    df_clean.to_parquet(output_path, index=True)
    
    print("✔ Deduplication complete!")
    
    # Verify no duplicates remain
    remaining_duplicates = df_clean.duplicated(subset=['InChI'], keep=False).sum()
    print(f"Remaining duplicates: {remaining_duplicates}")

def main():
    ap = argparse.ArgumentParser(description="Remove duplicate compounds from chemprop data")
    ap.add_argument("--input", default="chemprop_flat.parquet", help="input parquet file")
    ap.add_argument("--output", default="chemprop_flat_clean.parquet", help="output parquet file")
    args = ap.parse_args()
    
    deduplicate_chemprop(args.input, args.output)

if __name__ == "__main__":
    main() 