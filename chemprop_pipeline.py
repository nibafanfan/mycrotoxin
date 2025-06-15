# chemprop_pipeline.py
"""
Flatten a ChemProp JSON-lines dump into a tabular file that can be used
directly as model features (one column per property_token).

---------------------------------------------------------------------------
Input
    chemprop_api_raw.jsonl   – exactly what your batch API call returned
                               (one compound per line).

Output
    chemprop_flat.parquet    – index = compound index
                               columns:
                                   • name, cid, InChI   (metadata)
                                   • pred_<token>       (numeric prediction)
                                   • api_response_json  (raw API response)

The script is 100 % idempotent: if chemprop_flat.parquet already exists
and is newer than the JSONL, nothing happens – perfect for a DVC stage.

Usage
    python chemprop_pipeline.py  \
        --jsonl   chemprop_api_raw.jsonl \
        --out     chemprop_flat.parquet
---------------------------------------------------------------------------

Author: your-name-or-org
Created: 2025-06-14
"""
import argparse, json, pathlib, sys, pandas as pd

def flatten(jsonl_path: pathlib.Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open() as fh:
        for ln in fh:
            rec = json.loads(ln)
            row = {
                "index": int(rec["index"]),
                "name":  rec["name"],
                "cid":   rec.get("cid"),
                "InChI": rec["InChI"],
                "api_response_json": ln.strip(),  # keep raw JSON for category lookup
            }
            # one numeric column per assay/property
            for p in rec["api_response"]:
                tok = p["property_token"]              # e.g. 6173
                row[f"pred_{tok}"] = p["value"]        # model output
            rows.append(row)

    df = pd.DataFrame(rows).set_index("index").sort_index()
    # keep column order stable: metadata first, then predictions sorted
    meta_cols = ["name", "cid", "InChI", "api_response_json"]
    pred_cols = sorted(c for c in df.columns if c.startswith("pred_"))
    return df[meta_cols + pred_cols]

def main(jsonl: str, out: str) -> None:
    jsonl_path = pathlib.Path(jsonl).resolve()
    out_path   = pathlib.Path(out).resolve()

    # —— smart "do nothing if up-to-date" logic
    if out_path.exists() and out_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
        print(f"✔ {out_path.name} already up to date ({out_path.stat().st_size/1e6:.1f} MB)")
        return

    print("⏳ flattening …")
    df = flatten(jsonl_path)
    df.to_parquet(out_path, index=True)
    print(f"✔ saved {len(df):,} molecules  ×  {df.filter(like='pred_').shape[1]:,} features")
    print(f"→ {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="raw ChemProp JSONL")
    ap.add_argument("--out",   default="chemprop_flat.parquet", help="output Parquet")
    args = ap.parse_args()
    main(args.jsonl, args.out)
