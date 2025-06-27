#!/usr/bin/env python3
"""
reverse_map_features.py - Reverse map feature tokens to property names and categories

This script takes the top features from benchmark results and maps them back to
their actual property names, categories, and descriptions from the ChemProp API response.
"""
import argparse
import json
import pandas as pd
from pathlib import Path

def extract_property_mapping(df: pd.DataFrame) -> dict:
    """Extract property token to property info mapping from API response."""
    col = next(c for c in ("api_response_json", "api_response") if c in df)
    
    # Get the first non-null API response
    api_response = df[col].dropna().iloc[0]
    rec = json.loads(api_response)
    
    # Create mapping from property_token to full property info
    property_map = {}
    for prop in rec["api_response"]:
        token = prop["property_token"]
        property_map[token] = {
            "title": prop["property"]["title"],
            "categories": [cat["category"] for cat in prop["property"]["categories"]],
            "category_reasons": [cat.get("reason", "") for cat in prop["property"]["categories"]],
            "source": prop["property"].get("source", ""),
            "metadata": prop["property"].get("metadata", {}),
            "value": prop["value"]
        }
    
    return property_map

def map_top_features(parquet_path: str, top_features: list) -> pd.DataFrame:
    """Map top features to their property information."""
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    print("Extracting property mapping...")
    property_map = extract_property_mapping(df)
    
    def get_assay_description(prop_info):
        """Get the best available assay description from metadata."""
        metadata = prop_info["metadata"]
        
        # Try different possible fields based on source
        if "BioAssay Name" in metadata:
            return metadata["BioAssay Name"]
        elif "description" in metadata:
            return metadata["description"]
        elif "Assay" in metadata:
            return metadata["Assay"]
        elif "assay_type" in metadata:
            return metadata["assay_type"]
        elif "bao_format" in metadata:
            return f"{metadata.get('assay_type', '')} - {metadata.get('bao_format', '')}"
        else:
            return f"{prop_info['source']} - {prop_info['title']}"
    
    # Create results dataframe
    results = []
    for feature in top_features:
        if feature.startswith("pred_"):
            token = int(feature.replace("pred_", ""))
            if token in property_map:
                prop_info = property_map[token]
                assay_desc = get_assay_description(prop_info)
                results.append({
                    "feature": feature,
                    "token": token,
                    "title": prop_info["title"],
                    "categories": "; ".join(prop_info["categories"]),
                    "category_reasons": "; ".join(prop_info["category_reasons"])[:150] + "..." if len("; ".join(prop_info["category_reasons"])) > 150 else "; ".join(prop_info["category_reasons"]),
                    "source": prop_info["source"],
                    "assay_description": assay_desc[:100] + "..." if len(assay_desc) > 100 else assay_desc
                })
            else:
                results.append({
                    "feature": feature,
                    "token": token,
                    "title": "UNKNOWN",
                    "categories": "UNKNOWN",
                    "category_reasons": "UNKNOWN",
                    "source": "UNKNOWN",
                    "assay_description": "UNKNOWN"
                })
    
    return pd.DataFrame(results)

def main():
    # Top features from the benchmark results
    genotoxicity_features = [
        "pred_4344", "pred_3245", "pred_2982", "pred_4529", "pred_3506",
        "pred_4623", "pred_6139", "pred_4874", "pred_4562", "pred_5249"
    ]
    
    carcinogenicity_features = [
        "pred_3687", "pred_4715", "pred_4212", "pred_6042", "pred_4300",
        "pred_2687", "pred_3340", "pred_5980", "pred_2431", "pred_5778"
    ]
    
    mutagenicity_features = [
        "pred_3991", "pred_2434", "pred_2431", "pred_5836", "pred_4018",
        "pred_3135", "pred_3129", "pred_3761", "pred_3858", "pred_2872"
    ]
    
    parquet_path = "chemprop_flat_clean.parquet"
    
    print("=" * 100)
    print("GENOTOXICITY TOP FEATURES")
    print("=" * 100)
    gen_df = map_top_features(parquet_path, genotoxicity_features)
    print(gen_df.to_string(index=False, max_colwidth=50))
    
    print("\n" + "=" * 100)
    print("CARCINOGENICITY TOP FEATURES")
    print("=" * 100)
    car_df = map_top_features(parquet_path, carcinogenicity_features)
    print(car_df.to_string(index=False, max_colwidth=50))
    
    print("\n" + "=" * 100)
    print("MUTAGENICITY TOP FEATURES")
    print("=" * 100)
    mut_df = map_top_features(parquet_path, mutagenicity_features)
    print(mut_df.to_string(index=False, max_colwidth=50))
    
    # Save to CSV for easier viewing
    gen_df.to_csv("genotoxicity_top_features.csv", index=False)
    car_df.to_csv("carcinogenicity_top_features.csv", index=False)
    mut_df.to_csv("mutagenicity_top_features.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"- genotoxicity_top_features.csv")
    print(f"- carcinogenicity_top_features.csv")
    print(f"- mutagenicity_top_features.csv")

if __name__ == "__main__":
    main() 